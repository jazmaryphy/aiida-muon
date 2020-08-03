import numpy as np

# PyMatGen stuff
from pymatgen.core.structure import Structure
from pymatgen.analysis.defects.core import Interstitial
from pymatgen import MPRester, PeriodicSite
#####from pymatgen.core.sites import PeriodicSite
from pymatgen import Specie
from pymatgen.analysis.defects.generators import VacancyGenerator, SubstitutionGenerator, \
                                                VoronoiInterstitialGenerator, InterstitialGenerator
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer

####### Tess defect generator
##from aiida.aiida_muons.workchains.aiida_muon_tess import get_interstitials as tess_interstitials
from aiida_muon_tess import get_interstitials as tess_interstitials
####### UGRID
#from aiida.aiida_muons.workchains.aiida_muesr_ugrid import generate_uniform_grid, show_cell, show_structure_with_muon
from aiida_muesr_ugrid import generate_uniform_grid, show_cell, show_structure_with_muon
######
####### AiiDA stuff

######
from aiida.orm import Code, Str, Float
######from aiida.tools.dbimporters.plugins.cod import CodDbImporter
######from aiida.engine import run
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.engine import submit, run, launch


# Override or add parameters:
def merge(a, b, path=None):
    "merges b into a, from https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                print('Setup conflict at %s: you setting has been overwritten' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def grid_parameters(file_path,grid_size):
    return file_path, grid_size
    
def show_structures(bulk_structure):
    return show_cell(bulk_structure)
    
def show_structures_with_muon(bulk_structure,  mu_position,  sc_size):
    return show_structure_with_muon(bulk_structure,  mu_position,  sc_size )

def generate_supercells(bulk_structure, grid_param, supercell = (2, 2, 2), method='voronoi', positions=None):
    """
    This is a wrapper to various methods to generate interstitial impurities.
    """

    #magnetic_structure = CollinearMagneticStructureAnalyzer(bulk_structure, make_primitive=False)
    #if magnetic_structure.is_magnetic:
    #    #spin_bulk_structure = magnetic_structure.get_structure_with_spin()
    #    spin_bulk_structure = 
    #    spin_bulk_structure.make_supercell(supercell)
    site_properties_preservation = bulk_structure.copy()
    site_properties_preservation.make_supercell(supercell)
    
    structure_list = []

    ## Manually set ?
    if method == 'manual':
        if positions is None:
            raise ValueError('Position must be specified in manual mode!')
        r = []
        for position in positions:
            defect_site = PeriodicSite("H", position, bulk_structure.lattice)
            r.append(Interstitial(bulk_structure, defect_site, charge=0.))
    elif method == 'tess':
        r = []
        for position in tess_interstitials(bulk_structure):
            defect_site = PeriodicSite("H", position, bulk_structure.lattice, coords_are_cartesian=False)
            r.append(Interstitial(bulk_structure, defect_site, charge=0.))
    elif method == 'infit':
        r = list(InterstitialGenerator( bulk_structure, 'H'))
    elif method == 'ugrid':
         r = []
         bulkStructureGrid, grid_size=grid_param[0], grid_param[1]
         for position in generate_uniform_grid(bulkStructureGrid, grid_size):
             defect_site = PeriodicSite("H", position, bulk_structure.lattice, coords_are_cartesian=False)
             r.append(Interstitial(bulk_structure, defect_site, charge=0.))
    else:        
        r = list(VoronoiInterstitialGenerator( bulk_structure, 'H'))

    for i,v in enumerate(r):
        struct=v.generate_defect_structure(supercell=supercell)
        
        
        ## Preserve magnetic structure
        #if (magnetic_structure.is_magnetic):
        #    spins = np.zeros(struct.num_sites)
        #    # TODO: check non collinear...
        #    for site1 in spin_bulk_structure:
        #        for i, site2 in enumerate(struct):
        #            if np.allclose(site1.coords, site2.coords, atol = site2.position_atol):
        #                spins[i] = site1.specie.spin
        #                break
        #                
        #    struct.add_spin_by_site(spins)
            
        for site1 in site_properties_preservation:
            for i, site2 in enumerate(struct):
                if np.allclose(site1.coords, site2.coords, atol = site2.position_atol):
                    #spins[i] = site1.specie.spin
                    struct[i].properties = site1.properties
                    break

        # Remove symmetry
        struct.perturb(0.0001)
        struct.translate_sites(-1, 0.1 * np.random.random(3), frac_coords=False)
        if struct.is_valid():
            structure_list.append(struct)
    return structure_list



def analyze_pymatgen_structure(pymatgen_structure, mark_muon=True, moment_to_polarization=True):
    """
    Convert pymatgen structure to aiida, correctly setting magnetic structure
    and atoms name. Optionally labels muon as 'mu'.
    returns: structure with kind names according to magnetic labels.
             spin type: 1, non magnetic; 2, collinear; 4 non-collinear;
             dictionary with magnetic_elements_kind:
             for collinear case:
               {'Cr': {'Cr1': 1.0, 'Cr2': -1.0}}
             for non-collinear care, like this:
               {'Cr': {'Cr1': array([0.        , 1.73205081, 2.        ]),
                       'Cr2': array([-1.5      , -0.8660254,  2.       ]),
                       'Cr3': array([ 1.5      , -0.8660254,  2.       ]),
                       'Cr4': array([-1.5      ,  0.8660254,  2.       ]),
                       'Cr5': array([ 0.        , -1.73205081,  2.        ]),
                       'Cr6': array([1.5      , 0.8660254, 2.       ])}
                }

    """

    # analyze magnetism
    magnetic_structure = CollinearMagneticStructureAnalyzer(pymatgen_structure, make_primitive=False)

    has_spin = 0
    if magnetic_structure.is_magnetic:
        if magnetic_structure.is_collinear:
            has_spin=2
        else:
            has_spin=4
    else:
        has_spin=1

    # generate collinear spin structure
    if (has_spin in (1,2)):
        structure_with_spin = magnetic_structure.get_structure_with_spin()
    else:
        structure_with_spin = pymatgen_structure


    # collect magnetic elements by name. For each of them create kinds
    kind_values = []
    magnetic_elements_kinds = {}
    n_sites = len(structure_with_spin)
    for s_idx, site in enumerate(structure_with_spin):

        # check spin and element name. Information is in slightly different places
        spin = site.specie.spin              if has_spin in (1,2) else site.magmom.moment
        element = site.specie.element.symbol if has_spin in (1,2) else site.specie.symbol

        # scale moment according to max spin polarization
        if moment_to_polarization:
            if site.specie.block == 'p':
                spin /= 3.
            elif site.specie.block == 'd':
                spin /= 5.
            elif site.specie.block == 'f':
                spin /= 7.


        kind_name = None
        if not np.allclose(np.abs(spin), 0.0):
            # checks if element was already found to be magnetic in a previous site
            # otherwise return an empty dictionary to be filled with the information
            # of this site
            kinds_for_element = magnetic_elements_kinds.get(element, {})
            
            # If the spin of this site is for this element is the same we found
            # previously, just give it the same kind, otherwise add it as 
            # a new kind for this element type.
            for kind, kind_spin in kinds_for_element.items():
                if np.allclose (spin, kind_spin):
                    kind_name = kind
                    break
            else:
                kind_name = '{}{}'.format(element, len(kinds_for_element)+1)
                kinds_for_element[kind_name] = spin
            
            # store the updated list of kinds for this element in the full dictionary.
            magnetic_elements_kinds[element] = kinds_for_element

        kind_values.append(kind_name)

    # last value of element checked from for loop...yeah, no namespaces in python
    if mark_muon and element == 'H':
        kind_values[-1] = 'mu'
    # Finally return structre, type of magnetic order
    # and a dictionary for magnetic elements, where all the kinds are reported together
    # with the value of the spin.
    return (pymatgen_structure.copy(site_properties={'kind_name': kind_values}), has_spin, magnetic_elements_kinds )


##### def pymatgen_to_aiida(pymatgen_structure, StructureData, mark_muon=True, moment_to_polarization=True):
#####     # AAA THIS IS DEPRECATED!
#####     """
#####     Convert pymatgen structure to aiida, correctly setting magnetic structure
#####     and atoms name. Optionally labels muon as 'mu'.
#####     """
##### 
#####     # analyze magnetism
#####     magnetic_structure = CollinearMagneticStructureAnalyzer(pymatgen_structure, make_primitive=False)
##### 
#####     has_spin = 0
#####     if magnetic_structure.is_magnetic:
#####         if magnetic_structure.is_collinear:
#####             has_spin=2
#####         else:
#####             has_spin=4
#####     else:
#####         has_spin=1
#####         if mark_muon:
#####             values = [None,]*len(pymatgen_structure)
#####             values[-1] = 'mu'
#####             pymatgen_structure = pymatgen_structure.copy(site_properties={'kind_name': values})
#####         return (StructureData(pymatgen=pymatgen_structure), has_spin, {})
##### 
#####     # generate collinear spin structure
#####     if (has_spin == 2):
#####         structure_with_spin = magnetic_structure.get_structure_with_spin()
#####     else:
#####         structure_with_spin = pymatgen_structure
#####     
#####     # initial definition of cell (is PBC specified by default?)
#####     cell = structure_with_spin.lattice.matrix.tolist()
#####     aiida_structure = StructureData(cell=cell)
##### 
#####     # collect magnetic elements by name. For each of them create kinds
#####     magnetic_elements_kinds = {}
#####     n_sites = len(structure_with_spin)
#####     for s_idx, site in enumerate(structure_with_spin):
##### 
#####         # check spin and element name. Information is in slightly different places
#####         spin = site.specie.spin              if has_spin == 2 else site.magmom.moment
#####         element = site.specie.element.symbol if has_spin == 2 else site.specie.symbol
#####         if moment_to_polarization:
#####             if site.specie.block == 'p':
#####                 spin /= 3.
#####             elif site.specie.block == 'd':
#####                 spin /= 5.
#####             elif site.specie.block == 'f':
#####                 spin /= 7.
#####             
#####         
#####         kind_name = None
#####         if not np.allclose(np.abs(spin), 0.0):
#####             # checks if element was already found to be magnetic in a previous site
#####             # otherwise return an empty dictionary to be filled with the information
#####             # of this site
#####             kinds_for_element = magnetic_elements_kinds.get(element, {})
#####             
#####             # If the spin of this site is for this element is the same we found
#####             # previously, just give it the same kind, otherwise add it as 
#####             # a new kind for this element type.
#####             for kind, kind_spin in kinds_for_element.items():
#####                 if np.allclose (spin, kind_spin):
#####                     kind_name = kind
#####                     break
#####             else:
#####                 kind_name = '{}{}'.format(element, len(kinds_for_element)+1)
#####                 kinds_for_element[kind_name] = spin
#####             
#####             # store the updated list of kinds for this element in the full dictionary.
#####             magnetic_elements_kinds[element] = kinds_for_element
##### 
#####         # prepare to add site to AiiDA structure...
#####         inputs = {
#####             'symbols': [x.symbol for x in site.species_and_occu.keys()],
#####             'weights': [x for x in site.species_and_occu.values()],
#####             'position': site.coords.tolist()
#####         }
#####         # ...and also its kind, in case we chose one...
#####         if kind_name is not None:
#####             inputs['name'] = kind_name
#####         # or maybe is this the muon?
#####         if element == 'H' and s_idx == n_sites-1:
#####             inputs['name'] = 'mu'
#####         # ...and it's done for this one!
#####         aiida_structure.append_atom(**inputs)
#####     
#####     # Finally return structre, type of magnetic order (1 or 2 for the time being) 
#####     # and a dictionary for magnetic elements, where all the kinds are reported together
#####     # with the value of the spin.
#####     return (aiida_structure, has_spin, magnetic_elements_kinds )
    


def run_wc(index, name, struct, group, codename, pseudo_family, k_distance=0.4, scale_element_init_moment={}, input_namelists={}, mu_plus='false', code_runtime_options=None):
    """
    Runs a workchain to obtain relaxed position of muon 
    """

    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options

    def merge_dict_of_dicts(superdict):
        """Given two dicts, merge them into a new dict as a shallow copy. Goes away with python3"""
        z = {}
        for d in superdict.values():
            z.update(d)
        return z

    
    builder = WorkflowFactory('quantumespresso.pw.relax').get_builder()
    builder.relaxation_scheme = Str('relax')

    builder.metadata.label = "{}-{}".format(name, index)
    builder.metadata.description = "Muon site relaxation workchain for {}, with initial position {} {} {}".format(name, *struct[-1].coords.tolist())


    StructureData     = DataFactory("structure")
    #builder.structure = StructureData(pymatgen=struct)
    #builder.structure, nspin, magnetic_elements_kinds = pymatgen_to_aiida(struct, StructureData)
    labeled_structure, nspin, magnetic_elements_kinds = analyze_pymatgen_structure(struct)
    builder.structure = StructureData(pymatgen=labeled_structure)

    if nspin == 2:
        # Set all polarizations to 0.4
        for k in magnetic_elements_kinds.keys():
            for idx in magnetic_elements_kinds[k].keys():
                magnetic_elements_kinds[k][idx] = magnetic_elements_kinds[k][idx] * scale_element_init_moment.get(k, 1.)
    elif nspin == 4:
        raise NotImplemented("Non collinear case not implemented.")


    Dict = DataFactory('dict')
    parameters_dict = {
        'CONTROL': {
            'calculation': 'relax',
            'restart_mode': 'from_scratch',
        },
        'SYSTEM': {
            'ecutwfc': 80,               # before is 60
            'ecutrho': 800,              # before is 600
            'occupations':'smearing',
            'smearing': 'gaussian',
            'degauss' : 0.02,            # before is 0.005
            'nspin': nspin, #            'starting_magnetization': merge_dict_of_dicts(magnetic_elements_kinds)
        },
        'ELECTRONS': {
            'electron_maxstep': 600,
            'conv_thr'    : 1.e-7,
            'mixing_beta' : 0.30,
            'mixing_mode' : 'local-TF',
        }
    }
    if nspin == 2: 
        parameters_dict['SYSTEM']['starting_magnetization'] = merge_dict_of_dicts(magnetic_elements_kinds)
        
    mu_plus0=False
    if mu_plus=='true':  # crazy solution
        parameters_dict['SYSTEM']['starting_charge'] = {'mu': 0.6}
        parameters_dict['SYSTEM']['tot_charge'] = 1.

    if input_namelists:
        # Remove some stuff from input namelist
        input_namelists.pop('CONTROL',0)
        input_namelists.pop('K_POINTS',0)
        input_namelists.pop('CELL_PARAMETERS',0)
        input_namelists.pop('ATOMIC_FORCES',0)
    
        if 'SYSTEM' in input_namelists.keys():
            for kw in ('nat ntyp A B C cosAB cosAC cosBC nbnd tot_charge').split():
                r = input_namelists['SYSTEM'].pop(kw,'')
                if r:
                    print('Your setting for {} has been removed'.format(r))
                    
            for kw in input_namelists['SYSTEM'].keys():
                r = ''
                if 'celldm' in kw:
                    r = input_namelists['SYSTEM'].pop(kw)
                if 'starting_' in kw:
                    r = input_namelists['SYSTEM'].pop(kw)
                if r:
                    print('Your setting for {} has been removed'.format(r))
                    
    
        parameters_dict = merge(input_namelists, parameters_dict)

    parameters = Dict(dict=parameters_dict)

    builder.base.pseudo_family = Str(pseudo_family)

    #builder.base.kpoints_distance = Float(0.4)
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(builder.structure)
    kpoints.set_kpoints_mesh_from_density(k_distance, force_parity=False)
    kpoints.store()

    settings_dict={}
    num_k_points = np.prod(kpoints.get_kpoints_mesh()[0])
    if num_k_points==1:
        settings_dict={'gamma_only': True}
    else:
        settings_dict={'gamma_only': False}
    
    builder.base.kpoints = kpoints

    builder.base.pw.code = Code.get_from_string(codename)
    builder.base.pw.parameters = parameters

    #if hubbard_file:
    #    builder.base.pw.hubbard_file = hubbard_file
    
    # AAAA: automatic_parallelization does not work!!!
    automatic_parallelization = False
    if automatic_parallelization:
        automatic_parallelization = get_automatic_parallelization_options(1,  24*60*60-60*5)
        builder.base.automatic_parallelization = Dict(dict=automatic_parallelization)
    else:
        if code_runtime_options is None or code_runtime_options == '':
                                        # num machines, time, mpi
            default_options = get_default_options(1, 24*60*60-60*5, True)
            builder.base.pw.metadata.options = default_options
        else:
            exec_options = code_runtime_options.split('|')[0].split()
            default_options = get_default_options(int(exec_options[0]), int(exec_options[1]), True)
            builder.base.pw.metadata.options = default_options            
        if code_runtime_options is None or code_runtime_options == '':
            npool = np.min([4, num_k_points])
            settings_dict['cmdline'] = ['-nk', str(npool), '-ndiag', '1']
        else:
            parallel_options = code_runtime_options.split('|')[1]
            settings_dict['cmdline'] = parallel_options.strip().split()

    builder.base.pw.settings = Dict(dict=settings_dict)

    clean_workdir = False
    final_scf = False
    if clean_workdir:
        builder.clean_workdir = Bool(True)

    if final_scf:
        builder.final_scf = Bool(True)

    calc = submit(builder)

    if not (group is None):
        group.add_nodes(calc)
    return calc











def run_calculation(index, name, struct, group, codename, pseudo_family, k_distance=0.35, scale_element_init_moment={}, mu_plus=False, code_runtime_options=None):

    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options

    def merge_dict_of_dicts(superdict):
        """Given two dicts, merge them into a new dict as a shallow copy. Goes away with python3"""
        z = {}
        for d in superdict.values():
            z.update(d)
        return z

    code = Code.get_from_string(codename)
    builder = code.get_builder()


    builder.metadata.label = "{}-{}".format(name, index)
    builder.metadata.description = "Magnetic test {} - {}".format(name, index)


    StructureData     = DataFactory("structure")
    labeled_structure, nspin, magnetic_elements_kinds = analyze_pymatgen_structure(struct, mark_muon=False)
    builder.structure = StructureData(pymatgen=labeled_structure)

    if nspin == 2:
        # Set all polarizations to 0.4
        for k in magnetic_elements_kinds.keys():
            for idx in magnetic_elements_kinds[k].keys():
                magnetic_elements_kinds[k][idx] = magnetic_elements_kinds[k][idx] * scale_element_init_moment.get(k, 1.)
    elif nspin == 4:
        raise NotImplemented("Non collinear case not implemented.")


    Dict = DataFactory('dict')
    parameters_dict = {
        'CONTROL': {
            'calculation': 'relax',
            'restart_mode': 'from_scratch',
        },
        'SYSTEM': {
            'ecutwfc': 80,               # before is 60
            'ecutrho': 800,              # before is 600
            'occupations':'smearing',
            'smearing': 'gaussian',
            'degauss' : 0.02,            # before is 0.005
            'nspin': nspin, #            'starting_magnetization': merge_dict_of_dicts(magnetic_elements_kinds)
        },
        'ELECTRONS': {
            'electron_maxstep': 600,
            'conv_thr'    : 1.e-7,
            'mixing_beta' : 0.30,
            'mixing_mode' : 'local-TF',
        }
    }
    if nspin == 2: 
        parameters_dict['SYSTEM']['starting_magnetization'] = merge_dict_of_dicts(magnetic_elements_kinds)
        
    if mu_plus == True:
       parameters_dict['SYSTEM']['starting_charge'] = {'mu': 0.6}
       parameters_dict['SYSTEM']['tot_charge'] = 1.


    parameters = Dict(dict=parameters_dict)
    builder.parameters = parameters

    builder.pseudo_family = Str(pseudo_family)
    
    #builder.base.kpoints_distance = Float(0.4)
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(builder.structure)
    kpoints.set_kpoints_mesh_from_density(k_distance, force_parity=False)
    #kpoints.store()

    settings_dict={}
    num_k_points = np.prod(kpoints.get_kpoints_mesh()[0])
    if num_k_points==1:
        settings_dict={'gamma_only': True}
    else:
        settings_dict={'gamma_only': False}
    
    builder.kpoints = kpoints

    

    #if hubbard_file:
    #    builder.base.pw.hubbard_file = hubbard_file
    
    # AAAA: automatic_parallelization does not work!!!
    automatic_parallelization = False
    if automatic_parallelization:
        automatic_parallelization = get_automatic_parallelization_options(1,  24*60*60-60*5)
        builder.automatic_parallelization = Dict(dict=automatic_parallelization)
    else:
        if code_runtime_options is None or code_runtime_options == '':
                                        # num machines, time, mpi
            default_options = get_default_options(1, 24*60*60-60*5, True)
            builder.metadata.options = default_options
            #builder.metadata.options.resources = {'num_mpiprocs_per_machine': 64, 'tot_num_mpiprocs': 64}  # I am not sure what I am doing
        else:
            exec_options = code_runtime_options.split('|')[0].split()
            default_options = get_default_options(int(exec_options[0]), int(exec_options[1]), True)
            builder.metadata.options = default_options
            #builder.metadata.options.resources = {'num_mpiprocs_per_machine': 64, 'tot_num_mpiprocs': 64} # I am not sure what I am doing
        if code_runtime_options is None or code_runtime_options == '':
            npool = np.min(8, num_k_points)
            settings_dict['cmdline'] = ['-nk', str(npool), '-ndiag', '1']
        else:
            parallel_options = code_runtime_options.split('|')[1]
            settings_dict['cmdline'] = parallel_options.strip().split()

    builder.settings = Dict(dict=settings_dict)


    calc = submit(builder)
    
    if not (group is None):
        group.add_nodes(calc)
    return calc


def check_group_elements(group_name):
    from aiida.orm import Group # aiida stuff should not be here I think

    g = Group.get(label=group_name)
    for element in g.nodes:
        yield element


def run_hyperfine(uuid, group, pp_codename, projwfc_codename, code_runtime_options=None):  # I add name
    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options
    ##from aiida.aiida_muons.workchains.hf import PwContactFieldWorkChain
    from hf import PwContactFieldWorkChain
    from aiida.engine import submit, run, launch
    from aiida.orm import Group, load_node
    
    
    g, just_created = Group.objects.get_or_create(group)

    b = PwContactFieldWorkChain.get_builder()
    calc  = load_node(uuid)
    
    b.parent_folder = calc.outputs.remote_folder    

    b.pp.code = Code.get_from_string(pp_codename)
    b.projwfc.code = Code.get_from_string(projwfc_codename)

    b.pp.metadata.options=get_default_options(1, 2000, True) 
    b.projwfc.metadata.options=get_default_options(1, 2000, True) 

    
    hf  = submit(b)

    g.add_nodes(hf)

