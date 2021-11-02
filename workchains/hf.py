#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[1]:


import random
import numpy as np
from aiida import orm
###from aiida import load_profile
from pymatgen import Lattice, Structure
from aiida.common import AttributeDict, exceptions
from aiida.orm import  Int, List, Float, load_node
from aiida.engine import calcfunction, workfunction
from aiida.common.extendeddicts import AttributeDict
from aiida.orm.nodes.process.calculation.calcjob import CalcJobNode
from aiida.engine import WorkChain, ToContext, if_, while_, append_
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options


# In[2]:


###load_profile()

PwCalculation = CalculationFactory('quantumespresso.pw')
PpCalculation = CalculationFactory('quantumespresso.pp')
PjwfcCalculation = CalculationFactory('quantumespresso.projwfc')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


# In[3]:


def split_element_and_number(s):
    return (''.join(filter(str.isdigit, s)) or None,
            ''.join(filter(str.isalpha, s)) or None)

def validate_equiv_site_lists(sites, _):
    """Validate the equiv muon site lists in  input."""
    if sites and len(sites) < 1:
        return 'need at least 1 muon site.'


def validate_equiv_count(value, _):
    """Validate the `equiv_count` input."""
    if value is not None and value < 1:
        return 'need at least 1 equiv muon site.'

def create_amatch_structure(dict):
    #get_structure(dict):
    """We create a matching StructureData as stored in aiida
    Parameters:
              dict : dict
                    lattice: lattice matrix
                    species: specie types
                    positions: fractional positions (Converted to cartesian, this is a MUST)

    Returns: StructureData
    """
    lattice = dict['lattice']
    species = dict['species']
    positions = dict['positions']
    mupos = []
    structure = LatticeStructureData(lattice)
    for kind_name, pos in zip(species, positions):
        symbol  = split_element_and_number(kind_name)[-1]
        if symbol=='No':
            symbol='H'
            mupos.append(list(pos))
        structure.append_atom(position = np.dot(pos, lattice), symbols=symbol, name=kind_name)
    return structure, np.array(mupos)

def LatticeStructureData(lattice_matrix):
    """We create an empty Structure data,
    and later on we append the the species and their positions
    to match the aiida stored structure
    Parameters:
    -----------
              lattice_matrix : 3x3 matrix
    Returns: StructureData
    """
    pymatgen_structure = Structure(Lattice(lattice_matrix), [], [])
    StructureData = DataFactory("structure")
    return StructureData(pymatgen=pymatgen_structure)

def create_pymatgen_structure(dict):
    """To return a pymatgen structure
    """
    from pymatgen import Lattice, Structure

    lattice = Lattice(dict['lattice'])
    species = dict['species']
    positions = dict['positions']
    return Structure(lattice, species, positions)


# In[4]:


@calcfunction
def ParsePpPlot(pp_calc_out):
    import io
    outfile = pp_calc_out.open('aiida.out')
    outfile.seek(0)
    while True:
        l = outfile.readline()
        if 'Reading data from file  aiida.filplot' in l:
            break
        if l == '':
            raise RuntimeError("Invalid output file")
    f = io.StringIO()

    try:
        while True:
            l = outfile.readline()
            if 'Output format:' in l:
                break
            f.write(l)
            if l == '':
                raise RuntimeError("Invalid output file")
    except:
        raise RuntimeError("Unexpected error")
    finally:
        f.seek(0)

    from ase.io.cube import read_cube_data
    data, atoms = read_cube_data(f)
    return orm.Float(data[0,0,0])

@calcfunction
def MoveImpurityToOrigin(original_structure):
    """To translate the muon sites to origin. To make sure there is adequate grid of kpoints
    Paramters:
    ----------
             original_structure: relaxed structure of material containing a muon
    Returns:
    --------
           A structure with the muon position translated to the origin
    """
    ##### # ASE VERSIION
    ##### atms = original_structure.get_ase()
    ##### atms.translate(-atms[-1].position)
    ##### StructureData = DataFactory("structure")
    ##### return StructureData(ase=atms)

    # PYMATGEN
    atms = original_structure.get_pymatgen_structure()
    atms.translate_sites(range(atms.num_sites),-atms[-1].coords, frac_coords=False)
    StructureData = DataFactory("structure")
    return StructureData(pymatgen=atms)

@calcfunction
def SpinDensityToContactField(spin_up,spin_dn):
    """To get the hyperfine field at the muon site
    Parameters:
    -----------
               spin_up: spin up polarizations
               spin_dn: spin down polarizations
    Returns:
    --------
            Contact hyperfine field in Tesla
    """
    spin_up=orm.Float(spin_up.get_array('data')[0,0,0])
    spin_dn=orm.Float(spin_dn.get_array('data')[0,0,0])
    return 52.430351 * (spin_up - spin_dn) # In Tesla

@calcfunction
def SavePositions(positions):
    """
    """
    positions = np.array(positions.get_array('structure_positions'))
    PossArray = DataFactory('array')()
    PossArray.set_array('structure_positions', positions)
    return PossArray

@calcfunction
def SaveContactField(field_dict):
    """
    """
    field = field_dict.get_dict()
    dic = {}
    for key, value in field.items():
        dic[key] = value
    data_dict =  DataFactory('dict')(dict=dic)
    return data_dict


# In[5]:


class PwContactFieldWorkChain(WorkChain):
    """Workchain to relax a structure using Quantum ESPRESSO pw.x"""

    @classmethod
    def define(cls, spec):
        super(PwContactFieldWorkChain, cls).define(spec)

        spec.expose_inputs(PpCalculation, namespace='pp', exclude=('parent_folder','parameters'))
        spec.expose_inputs(PjwfcCalculation, namespace='projwfc', exclude=('parent_folder','parameters'))

#         spec.expose_inputs(PwBaseWorkChain, namespace='scf', exclude=('clean_workdir', 'pw.structure'))
#         spec.input('equiv_sites', valid_type=orm.List, required=False, validator=validate_scale_factors, help='the list of muon equiv site.')
#         spec.input('equiv_sites', valid_type=orm.List, help='the list of muon equiv site.')
#         spec.input('structures', valid_type=orm.Dict, help='a dictionary of structures to use')
#         spec.input('structure', valid_type=orm.StructureData, help='The pristine supercell pymatgen structure')
#         spec.input('equiv_sites', valid_type=orm.ArrayData, help='the list of muon equiv site.')

        spec.input('equiv_sites', valid_type=orm.Dict, help='a dictionary of structures to use')
        """
        # equiv_sites contains a collection of nth dictionary  (of equivalent muon sites)
        # each contains a lattice parameters (3x3 matrix), a list of species and nd.arrayx3
        # of ion positions. Why do that?
        # parsing a dict of pymatgen structure is not JSON serializable
        # better if we create the structures using 'create_pymatgen_structure' along the way
        # by parsing the lattice, species and positions of ions (including the muon in) in
        # 'create_pymatgen_structure(lattice, species, positions)' function
        """
        spec.input('equiv_count', valid_type=orm.Int, default=lambda: orm.Int(1),
                   help='number of structures to use')
        spec.input('parent_folder', valid_type=orm.RemoteData)

        spec.input('meta_convergence', valid_type=orm.Bool, default=lambda: orm.Bool(True))
        #spec.input('meta_convergence', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('max_meta_convergence_iterations', valid_type=orm.Int, default=lambda: orm.Int(5))
        spec.input('contact_field_convergence', valid_type=orm.Float, default=lambda: orm.Float(0.1))
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))


        spec.outline(
            cls.setup_init,
            while_(cls.should_refine_contact_field)(
                cls.run_init_scf,
                cls.inspect_init_scf,
                cls.run_pp_up,
                cls.run_pp_dn,
                cls.inspect_pp,
            ),
            cls.run_projwfc,
            cls.results,
            if_(cls.equiv_site_is_greater_than_one)(
                cls.next_workchain,
                while_(cls.run_next_workchain)(
                    cls.setup_next_workchain,
                    while_(cls.should_refine_contact_field)(
                        cls.scf_next_workchain,
                        cls.inspect_scf_next_workchain,
                        cls.run_pp_up,
                        cls.run_pp_dn,
                        cls.inspect_pp,
                    ),
                    cls.extract_contact_field_and_structure,
                    #check if all struct done!
                    cls.final_calc,
                ),
                cls.finalize,
            ),

        )

        spec.exit_code(0, 'NO_ERROR', message='done with calculations, everything works!')
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='the scf PwBasexWorkChain sub processes did not finish successfully.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='the relax PwBaseWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_FINAL_SCF',
            message='the final scf PwBaseWorkChain sub process failed')
        #spec.expose_outputs(PwBaseWorkChain)

        spec.output('contact_field', valid_type=orm.Float,
                     help='The contact field in Tesla at each calculations.')

#         spec.output('structure_contact_field_dict', valid_type=DataFactory('array'),
#                     help='The list containing all the contact field and structures')
#         spec.output('structure_contact_field_dict', valid_type=orm.Dict,
#                     help='A dictionary containing all the contact field and structures')

        spec.output('contact_field_dict', valid_type=orm.Dict,
                    help='A dictionary containing all the contact field for each structure')
        spec.output('structure_positions', valid_type=DataFactory('array'),
                     help='a list containing the nuclei positions (with mu) for each structures')

        spec.expose_outputs(PjwfcCalculation, namespace='projwfc')
        spec.expose_outputs(PwCalculation, namespace='pw')


    def get_last_calc(self):
        """Get a last calc node"""
        self.last_calc = self.inputs.parent_folder.creator

    def get_workchain_builder(self):
        """Return the builder from original relax workchain."""
        last_calc = self.inputs.parent_folder.creator
        return last_calc.get_builder_restart()

    def get_current_folder(self):
        """get current folder specify from inputs"""
        return self.inputs.parent_folder

    def get_original_relax_structure(self):
        """Return relax structure for given Node in pymatgen Structure form"""
        last_calc = self.inputs.parent_folder.creator #self.inputs.parent_folder.creator
        return last_calc.outputs.output_structure

    def get_pymatgen(self):
        """get structure in pymatgen form
        """
        return self.get_original_relax_structure().get_pymatgen_structure()

    def update_builder_parameters(self):
        #get_pw_parameters(self):
        """to change some original relax calculation parameters"""
        restart_builder = self.get_workchain_builder()
        cmdline_settings = restart_builder.settings.get_dict()
        parameters = restart_builder.parameters.get_dict()
        parameters['CONTROL']['calculation'] = 'scf'
        parameters['CONTROL']['restart_mode'] = 'from_scratch'
        parameters['SYSTEM']['ecutwfc'] *= (1.0 + 0.1 * self.ctx.iteration) # 1.1
        parameters['SYSTEM']['ecutrho'] *= (1.0 + 0.1 * self.ctx.iteration) # 1.1
        #parameters['ELECTRONS']['conv_thr'] = 1.0e-2
        #self.ctx.restart_builder.parameters = orm.Dict(dict=parameters)
        return parameters

    def setup_init(self):
        """Define structure and builder in the context to be the input structure."""
        self.ctx.current_parent_folder = self.get_current_folder()
        original_structure = self.get_original_relax_structure()
        self.ctx.restart_builder = self.get_workchain_builder()
        self.ctx.restart_builder.structure = original_structure

        self.ctx.restart_builder.structure = MoveImpurityToOrigin(original_structure)

        # Continue to submit workchains until this is True
        self.ctx.current_contact_field = None
        self.ctx.is_converged = False
        self.ctx.iteration = 0

    def should_refine_contact_field(self):
        """
        Return whether a SCF+PP workchain should be run, which is the case as long as the contact field
        change between two consecutive relaxation runs is larger than the specified volume convergence
        threshold value and the maximum number of meta convergence iterations is not exceeded
        """
        return not self.ctx.is_converged and self.ctx.iteration < self.inputs.max_meta_convergence_iterations.value

    def run_init_scf(self):
        """
        """
        self.ctx.iteration += 1
        parameters = self.update_builder_parameters()
        self.ctx.restart_builder.parameters = orm.Dict(dict=parameters)
        running = self.submit(self.ctx.restart_builder)
        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(running.pk, 'scf'))
        #return self.to_context(workchain_init_scf=append_(running))
        return ToContext(workchain_init_scf=append_(running))

    def inspect_init_scf(self):
        """Check that the first workchain finished successfully or abort the workchain."""
        workchains = self.ctx.workchain_init_scf[-1]
        self.ctx.current_parent_folder = workchains.outputs.remote_folder

    def run_pp_up(self):
        """ To run a post-processing to compute spin-up contribution of spin density
        """
        return self.run_pp('up')

    def run_pp_dn(self):
        """To run a post-processing to compute spin-down contribution of spin density
        """
        return self.run_pp('dn')

    def run_pp(self, spin):
        """The inputs to run the post-processing calculations
        Parameters:
        -----------
                  spin:
                      'up'=for spin-up calculations
                      'dn'=for spin-down calculations
        Returns:
        --------
        """

        #PpCalculation = CalculationFactory('quantumespresso.pp')
        inputs = self.exposed_inputs(PpCalculation, namespace='pp')
        inputs['parent_folder'] = self.ctx.current_parent_folder

        inputs['parameters'] = orm.Dict(dict={
                                                'INPUTPP': {
                                                'plot_num': 17,
                                                'spin_component': 1 if spin=='up' else 2
                                                },
                                                'PLOT' : {
                                                'iflag' : 3
                                                }
                                        })

        running = self.submit(PpCalculation, **inputs)

        self.report('launching PPCalculation<{}>'.format(running.pk))

        return ToContext(workchains=append_(running))
        #return self.to_context(workchains=append_(running))

    def inspect_pp(self):
        """
        Compare the contact field of the relaxed structure of the last completed workchain with the previous.
        If the difference ratio is less than the volume convergence threshold we consider the cell relaxation
        converged and can quit the workchain.
        """

        pp_up = self.ctx.workchains[-2]
        pp_dn = self.ctx.workchains[-1]

        if not (pp_up.is_finished_ok and pp_dn.is_finished_ok):
            self.report('PP failed with exit status {} {}'.format(pp_up.exit_status, pp_dn.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        # current sync version requires this
        spin_up = pp_up.outputs.output_data.get_array('data')[0,0,0] #ParsePpPlot(pp_up.outputs.retrieved)
        spin_dn = pp_dn.outputs.output_data.get_array('data')[0,0,0] #ParsePpPlot(pp_dn.outputs.retrieved)


        prev_contact_field = self.ctx.current_contact_field
        curr_contact_field = SpinDensityToContactField(pp_up.outputs.output_data, pp_dn.outputs.output_data)

        self.report('after iteration {} contact field of relaxed structure is {}'
                    .format(self.ctx.iteration, curr_contact_field))

        # After first iteration, simply set the contact field and restart the next base workchain
        if not prev_contact_field:
            self.ctx.current_contact_field = curr_contact_field

            # If meta convergence is switched off we are done
            if not self.inputs.meta_convergence.value:
                self.ctx.is_converged = True
            return

        # Check whether the contact field is converged
        curr_contact_threshold = self.inputs.contact_field_convergence.value
        if abs(prev_contact_field.value) > 0.0000000001: # replace with EPS
            contact_field_difference = abs(prev_contact_field.value - curr_contact_field.value) / prev_contact_field.value
        else:
            self.report('Warning, previous contact field was 0!? Running again')
            contact_field_difference = curr_contact_threshold*1.1


        if contact_field_difference < curr_contact_threshold:
            self.ctx.is_converged = True
            self.report('relative contact field difference {} smaller than convergence threshold {}'
                .format(contact_field_difference, curr_contact_threshold))
        else:
            self.report('current relative contact field difference {} larger than convergence threshold {}'
               .format(contact_field_difference, curr_contact_threshold))

        self.ctx.current_contact_field = curr_contact_field

        return

    def run_projwfc(self):

        inputs = self.exposed_inputs(PjwfcCalculation, namespace='projwfc')
        inputs['parent_folder'] = self.ctx.current_parent_folder

        inputs['parameters'] = orm.Dict(dict={
                                                'PROJWFC': {
                                                    'DeltaE' : 0.2,
                                                    'ngauss' : 1,
                                                    'degauss' : 0.02
                                                }})

        running = self.submit(PjwfcCalculation, **inputs)

        self.report('launching ProjWfcCalculation<{}>'.format(running.pk))

        return ToContext(workchains=append_(running))
        #return self.to_context(workchains=append_(running))

    def results(self):
        """Attach the output parameters and structure of the last workchain to the outputs."""
        if self.ctx.is_converged and self.ctx.iteration <= self.inputs.max_meta_convergence_iterations.value:
            self.report('workchain completed after {} iterations'.format(self.ctx.iteration))
        else:
            self.report('maximum number of meta convergence iterations exceeded')


        projwfc = self.ctx.workchains[-1]
        pw = self.ctx.current_parent_folder.creator

        self.out_many(self.exposed_outputs(pw, PwCalculation, namespace='pw'))
        self.out_many(self.exposed_outputs(projwfc, PjwfcCalculation, namespace='projwfc'))
        self.out('contact_field', self.ctx.current_contact_field)

    def equiv_site_is_greater_than_one(self):
        """condition to run equiv site"""
        return self.inputs.equiv_count.value > 1

    def get_equiv_muon_sites(self):
        """Return the list of muon sites"""
        if 'equiv_sites' in self.inputs:
           #return self.inputs.equiv_sites.get_array("sites")
           return self.inputs.equiv_sites.get_dict()

    def next_workchain(self):
        """Initial worchain for equiv. positions"""
        self.ctx.next_calc = 0
        self.ctx.is_finished = False
        # Define the contact field dictionary list
        self.ctx.positions_array = []
        self.ctx.contact_field_dict = {}

    def run_next_workchain(self):
        """
        Return whether a new workchain should be run.
        This is the case as long as the last workchain has not finished successfully.
        """
        return not self.ctx.is_finished

    def setup_next_workchain(self):
        """init iteration parameters
        """
        self.ctx.current_parent_folder = self.get_current_folder()
        self.ctx.structure_dict = self.get_equiv_muon_sites()                          #self.site_to_dict() # structures
        list_sites = list(self.ctx.structure_dict.keys())
        self.ctx.label = list_sites[self.ctx.next_calc]
        self.ctx.this_structure0 = self.ctx.structure_dict[self.ctx.label]             # structure with muon
        """
        # Since the index of the positions of ions are distinct and not change in QE.
        # We can match the structure and parse all the available prevous data.
        """
        self.ctx.this_structure, self.ctx.mupos = create_amatch_structure(self.ctx.this_structure0)
        self.ctx.inputs = self.get_workchain_builder()
        self.ctx.inputs.structure = self.ctx.this_structure
        self.ctx.inputs.structure.store()
        self.ctx.inputs.structure = MoveImpurityToOrigin(self.ctx.this_structure)
        self.ctx.current_contact_field = None
        self.ctx.is_converged = False
        self.ctx.iteration = 0

    def scf_next_workchain(self):
        """Initialize and Run the next workchain"""
        self.ctx.iteration += 1
        parameters = self.update_builder_parameters()
        self.ctx.inputs.parameters = orm.Dict(dict=parameters)
        self.ctx.inputs.parameters.store()
        running = self.submit(self.ctx.inputs)
        #running = self.submit(self.ctx.restart_builder)
        self.report('running PwBaseWorkChain<{}> calculation for equiv. {} with position {}'.
                    format(running.pk, self.ctx.label, self.ctx.mupos))
        #return self.to_context(workchain_equiv=append_(running))
        return ToContext(workchain_equiv=append_(running))

    def inspect_scf_next_workchain(self):
        """Inspect the workchain for any errors"""
        workchains = self.ctx.workchain_equiv[-1]
        self.ctx.current_parent_folder = workchains.outputs.remote_folder

    def extract_contact_field_and_structure(self):
        """Extract contact field and the structure used"""
        # store both contact field and structure
        label = str(self.ctx.label)
        positions = self.ctx.this_structure0['positions']
        self.ctx.positions_array.append(positions)
        self.ctx.contact_field_dict[label] = self.ctx.current_contact_field.value

    def on_terminated(self):
        """
        If the clean_workdir input was set to True, recursively collect all called Calculations by
        ourselves and our called descendants, and clean the remote folder for the JobCalculation instances
        """
        super(PwContactFieldWorkChain, self).on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report('cleaned remote folders of calculations: {}'.format(' '.join(map(str, cleaned_calcs))))

    def final_calc(self):
        """Final calculations"""
        if self.ctx.next_calc == self.inputs.equiv_count.value-1: self.ctx.is_finished = True
        self.ctx.next_calc +=1

    def finalize(self):
        """
        Finalize the workchain.
        Take the contact field container and set is as an output of this workchain.
        """
        # Due to data provenance we cannot return AiiDA data containers that have
        # not been passed through a calcfunction, workfunction or a workchain. Create this now.
        #all_contact_field = store_all_contact_field(DataFactory('list')(list=self.ctx.all_contact_field))

        #dict_data = DataFactory('dict')(dict=self.ctx.structure_contact)

        field_dict = DataFactory('dict')(dict=self.ctx.contact_field_dict)
        positions = np.array(self.ctx.positions_array)
        PossArray = DataFactory('array')()
        PossArray.set_array('structure_positions', positions)

        # And then store the output on the workchain
        #self.out('structure_contact_field_dict', dict_data)
        self.out('contact_field_dict', SaveContactField(field_dict))
        self.out('structure_positions', SavePositions(PossArray))
