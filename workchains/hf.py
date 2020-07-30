# -*- coding: utf-8 -*-
from aiida.common.extendeddicts import AttributeDict
from aiida import orm
#from aiida.orm.data.base import Bool, Float, Int, Str
#from aiida.orm.data.structure import StructureData
from aiida.orm.nodes.process.calculation.calcjob import CalcJobNode


from aiida.common import AttributeDict, exceptions
from aiida.engine import calcfunction, workfunction
from aiida.engine import WorkChain, ToContext, if_, while_, append_
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options

PwCalculation = CalculationFactory('quantumespresso.pw')
PpCalculation = CalculationFactory('quantumespresso.pp')
PjwfcCalculation = CalculationFactory('quantumespresso.projwfc')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


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
def SpinDensityToContactField(spin_up,spin_dn):
    return 52.430351 * (spin_up-spin_dn) # 52.430351 T

@calcfunction
def MoveImpurityToOrigin(original_structure):
    ##### # ASE VERSIION
    ##### atms = original_structure.get_ase()
    ##### atms.translate(-atms[-1].position)
    ##### StructureData = DataFactory("structure")
    ##### return StructureData(ase=atms)
    
    # PYMATGEN
    atms = original_structure.get_pymatgen_structure()
    atms.translate_sites(range(atms.num_sites),-atms[-1].coords, frac_coords=False)
    StructureData     = DataFactory("structure")
    return StructureData(pymatgen=atms)



class PwContactFieldWorkChain(WorkChain):
    """Workchain to relax a structure using Quantum ESPRESSO pw.x"""

    @classmethod
    def define(cls, spec):
        super(PwContactFieldWorkChain, cls).define(spec)
        spec.expose_inputs(PpCalculation, namespace='pp', exclude=('parent_folder','parameters'))
        spec.expose_inputs(PjwfcCalculation, namespace='projwfc', exclude=('parent_folder','parameters'))

        spec.input('parent_folder', valid_type=orm.RemoteData)

        spec.input('meta_convergence', valid_type=orm.Bool, default=orm.Bool(True))
        spec.input('max_meta_convergence_iterations', valid_type=orm.Int, default=orm.Int(5))
        spec.input('contact_field_convergence', valid_type=orm.Float, default=orm.Float(0.1))
        spec.input('clean_workdir', valid_type=orm.Bool, default=orm.Bool(False))

        spec.outline(
            cls.setup,
            while_(cls.should_refine_contact_field)(
                cls.run_scf,
                cls.inspect_scf,
                cls.run_pp_up,
                cls.run_pp_dn,
                cls.inspect_pp,
            ),
            cls.run_projwfc,
            cls.results,
        )
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='the relax PwBaseWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_FINAL_SCF',
            message='the final scf PwBaseWorkChain sub process failed')
        #spec.expose_outputs(PwBaseWorkChain)
        spec.output('contact_field', valid_type=orm.Float)
        spec.expose_outputs(PjwfcCalculation, namespace='projwfc')
        spec.expose_outputs(PwCalculation, namespace='pw')

    def setup(self):
        """
        Input validation and context setup
        """
        last_calc = self.inputs.parent_folder.creator
        self.ctx.current_parent_folder = self.inputs.parent_folder
        self.ctx.restart_builder = last_calc.get_builder_restart() # get restart from remote folder
        self.ctx.restart_builder.structure = last_calc.outputs.output_structure
        
        # Adapt structure
        original_structure = last_calc.outputs.output_structure
        self.ctx.restart_builder.structure = MoveImpurityToOrigin(original_structure)
        
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


    def run_scf(self):
        """
        Run the PwBaseWorkChain to run a relax PwCalculation
        """
        self.ctx.iteration += 1

        #restart_builder = self.ctx.base_calculation.get_builder_restart()

        # increase convergence
        parameters = self.ctx.restart_builder.parameters.get_dict()
        parameters['CONTROL']['calculation'] = 'scf'
        parameters['CONTROL']['restart_mode'] = 'from_scratch'
        parameters['SYSTEM']['ecutwfc'] *= 1.1 # (1.0 + 0.1 * self.ctx.iteration)
        parameters['SYSTEM']['ecutrho'] *= 1.1 # (1.0 + 0.1 * self.ctx.iteration)
        
        self.ctx.restart_builder.parameters = orm.Dict(dict=parameters)

        #self.ctx.restart_builder.structure = MoveImpurityToOrigin(restart_builder.structure)

        running = self.submit(self.ctx.restart_builder)
        self.report('launching PwBaseWorkChain<{}>'.format(running.pk))
        return ToContext(workchains=append_(running))


    def inspect_scf(self):
        self.ctx.current_parent_folder = self.ctx.workchains[-1].outputs.remote_folder

    def run_pp_up(self):
        return self.run_pp('up')

    def run_pp_dn(self):
        return self.run_pp('dn')

    def run_pp(self, spin):

        #PpCalculation = CalculationFactory('quantumespresso.pp')
        inputs = self.exposed_inputs(PpCalculation, namespace='pp')
        inputs['parent_folder'] = self.ctx.current_parent_folder

        inputs['parameters'] = orm.Dict(dict={
                                                'inputpp': {
                                                'plot_num': 17,
                                                'spin_component': 1 if spin=='up' else 2
                                                },
                                                'plot' : {
                                                'output_format': 6,
                                                'iflag' : 3
                                                }
                                        })

        running = self.submit(PpCalculation, **inputs)

        self.report('launching PPCalculation<{}>'.format(running.pk))

        return ToContext(workchains=append_(running))


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
        spin_up = ParsePpPlot(pp_up.outputs.retrieved)
        spin_dn = ParsePpPlot(pp_dn.outputs.retrieved)


        prev_contact_field = self.ctx.current_contact_field
        curr_contact_field = SpinDensityToContactField(spin_up, spin_dn)

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
