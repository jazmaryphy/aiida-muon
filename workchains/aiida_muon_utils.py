#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import numpy as np
from pymatgen import Specie
from pymatgen import MPRester, PeriodicSite
from aiida.engine import submit, run, launch
from pymatgen.core.structure import Structure
from aiida.orm import Code, Str, Float, Bool, Group, List, Int
from aiida_muesr_ugrid import generate_uniform_grid
from pymatgen.analysis.defects.core import Interstitial
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.analysis.defects.generators import VacancyGenerator, SubstitutionGenerator,                                                  VoronoiInterstitialGenerator, InterstitialGenerator

######from aiida.tools.dbimporters.plugins.cod import CodDbImporter
######from aiida.engine import run
####### Tess defect generator
######from aiida_muon_tess import get_interstitials as tess_interstitials


# In[2]:


# Merge dictionary into a new dictionary 
def merge_dict_of_dicts(superdict):
    """Given two dicts, merge them into a new dict as a shallow copy. Goes away with python3"""
    z = {}
    for d in superdict.values():
        z.update(d)
    return z
    

# Override or add parameters:
def merge(a, b, path=None):
    """This function merges a python dictionary b into a and also 
    allows the option to update the values
    
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    
    Parameters
    ----------
    a : dict
        a dictionary to merge into
    b : dic
        a dictionary to merge
    Returns
    -------
    a : dict
    """
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


# In[3]:


# Split and element and its index
def split_element_and_number(s):
    return (''.join(filter(str.isdigit, s)) or None,
            ''.join(filter(str.isalpha, s)) or None)


#Prepare Hubbard parameters for each distinct species
def _distinct_species(distinct_species):
    """
    """
    from collections import Counter, defaultdict
    def flatten(A):
        rt = []
        for i in A:
            if isinstance(i,list): rt.extend(flatten(i))
            else: rt.append(i)
        return rt
    
    def duplicates(lst):
        cnt= Counter(lst)
        return [key for key in cnt.keys() if cnt[key]> 1]
    
    def duplicates_indices(lst, items= None):
        items, ind= set(lst) if items is None else items, defaultdict(list)
        for i, v in enumerate(lst):
            if v in items: ind[v].append(i)
        return ind
               
    distinct_species = flatten(distinct_species)
    distinct_species_symbols = []
    distinct_species_index0 = []
    distinct_species_index1 = []
    for i, s in enumerate(distinct_species):
        distinct_species_index0.append(i)
        distinct_species_index1.append(split_element_and_number(s)[0])
        distinct_species_symbols.append(split_element_and_number(s)[1])
   
    duplicates_iden = duplicates_indices(distinct_species_symbols, 
                                         duplicates(distinct_species_symbols)
                                        )
    duplicates1 = list(duplicates_iden.values()) 
    duplicates1 = [item for sublist in duplicates1 for item in sublist]
    remove_index = []
    for s in duplicates1:
        if distinct_species_index1[s] is None:
            remove_index.append(s) 
    for r in remove_index:
        del distinct_species[r]
    return distinct_species

def set_hubbard_u_parameters(specie_kinds, hubbard_dict):
    """Set Hubbard U parameters for each kind of specie

    Parameters
    ----------
    specie_kinds : list
        A list of specie types
    hubbard_dict : dict
        A dictionary containing species and its hubbard U parameters
    Returns
    -------
    A dictionary of hubbard U species and their values
    """    
    hubbard_u = {}
    for s in specie_kinds:
        sp_kind = split_element_and_number(s)[1]
        if sp_kind in hubbard_dict:
            hubbard_u[s] = hubbard_dict[sp_kind]
    return hubbard_u


# In[4]:


def check_group_elements(group_name):
    """To check group name on AiiDA database
    
    Parameters
    ----------
    group_name : str
        Group name label
    """
    from aiida.orm import Group # aiida stuff should not be here I think

    g = Group.get(label=group_name)
    for element in g.nodes:
        yield element

# Generate a single structure dictionary
def single_structure_dict(uuid):
    """Generate a structure dictionary purposeful for hyperfine calculations
    of only single muon stopping site.
    
    Parameter
    ---------
    uiid : int
        An AiiDA calculation identifier
    Returns : dic
    """
    calc=load_node(uuid)
    species = calc.outputs.output_trajectory.get_array("atomic_species_name")
    outputs = calc.outputs.output_structure.get_pymatgen_structure()
    Dic = {}
    Dic['site0'] = {}
    Dic['site0']['lattice'] = outputs.lattice.matrix
    Dic['site0']['species'] = species
    Dic['site0']['positions'] = outputs.frac_coords
    return Dic

def yes_or_no(question):
    """ User interactive question
    
    Parameter
    ---------
    question : str
        Question ask if yes (y) or no (n) to proceed
    Return : bool
    """
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False


# In[5]:


def generate_supercells(
    bulk_structure, 
    grid_size = 4, 
    supercell = (2, 2, 2), 
    method = 'grid', 
    positions=None
):
    """
    This is a wrapper to various methods to generate different interstitial impurities.
    
    Parameters
    ----------
    bulk_structure : Structure object
        A unitcell (magnetic or not) structure in form of Pymatgen Structure object 
    grid_size : int
        A grid size in all lattice directions to generate uniform 
        grid for interstitial impurities implemented by Pietro Bonfa. Default=4.
    supercell : tuple, numpy.ndarray
        Supercell size in a, b and c direction. Default = (2, 2, 2)
    method : str
        The method to generate interstial impurities. Default='grid'
    positions : None
        A numpy.ndarray of muon positions if `method="manual"`.
    
    Returns
    -------
    structure_list : list
        A list of supercell structure containing interstitial impurities
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
            defect_site = PeriodicSite("H", position, bulk_structure.lattice, coords_are_cartesian=False)
            r.append(Interstitial(bulk_structure, defect_site, charge=0.))
    elif method == 'tess':
        r = []
        for position in tess_interstitials(bulk_structure):
            defect_site = PeriodicSite("H", position, bulk_structure.lattice, coords_are_cartesian=False)
            r.append(Interstitial(bulk_structure, defect_site, charge=0.))
    elif method == 'infit':
        r = list(InterstitialGenerator( bulk_structure, 'H'))
    elif method == 'grid':
        r = []
        for position in generate_uniform_grid(bulk_structure, grid_size):
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
        if method != 'manual':
            # Here we assume, manual muon site are perturb.
            # Performs a random perturbation of the sites in the structure 
            # to break symmetries. A distance of 1e-4 Angs. surely does nothing
            # However we can perturb the muon site with a random vector with
            # 0.1 Angs. distance
            struct.perturb(distance=0.0001)
            struct.translate_sites(-1, 0.1 * np.random.random(3), frac_coords=False)           
        if struct.is_valid():
            structure_list.append(struct)
    return structure_list


# In[6]:


def unit_vector(vector):
    """ Returns the unit vector of the vector.
    Parameters:
    ----------
    vector: array
            3D vector
          
    Returns: 
    --------
    A unit vector
    """
    return vector / np.linalg.norm(vector)


def check_angle_bool(angle, angle_tolerance=10.0):
    """A function that returns boolean T/F if the angles is within the threshold
    Parameters:
    -----------
    angle: float
           the list of angles
    angle_tolerance: float
                     the minimum angle in degrees. Default=10.
    """
    if 0.0 <= angle <= angle_tolerance or 180.0-angle_tolerance <= angle <= 180.0+angle_tolerance:
        return True
    return False


def check_and_force_collinear(mypatstructure, angle_tolerance=10.0):
    """Make sure the list of magmoms use the same spin axis by taking the
    largest magnetic moments as a reference axis for collinear calculations
    
    Parameters:
    -----------
    mypatstructure: Structure
                    structure object
    angle_tolerance: float
                     the minimum angle to consider between the reference spin axis and
                     magnetic moments vector. Default=10.
    Returns:
    --------
           all_true : boolean, if True the angle is within the threshold
    """
    from pymatgen.electronic_structure.core import Magmom
    from pymatgen.util.coord import get_angle

    magmoms = mypatstructure.site_properties['magmom']
    cif_moments, direction = Magmom.get_consistent_set_and_saxis(magmoms)
    #print("direction", direction)
    magmoms_ = np.empty([0,3])
    for comp in magmoms:
        magmoms_ = np.vstack([magmoms_, [comp[0], comp[1], comp[2]]])
        #print("comp", comp[0], comp[1], comp[2])

    # filter non zero moments
    magmoms_nzr = [m for m in magmoms_ if abs(np.any(m))]
    true_or_false = Magmom.are_collinear(magmoms)
    #if true_or_false == False:

    angles = np.empty([0, 1])
    for m in magmoms_nzr:
        angles = np.vstack([angles, get_angle(unit_vector(m), direction, units="degrees")])
        #print("angles", angles)

    store_check = []
    for angle in angles:
        store_check.append(check_angle_bool(angle, angle_tolerance))

    all_true = np.allclose(True, store_check)
    return all_true


# In[7]:


def analyze_pymatgen_structure(
    pymatgen_structure, 
    mark_muon=True, 
    moment_to_polarization=True
):
    """
    Convert pymatgen structure to aiida usable, correctly setting magnetic structure
    and atoms name. Optionally labels muon as 'No'.
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
    
    Parameters
    ----------
    pymatgen_structure : Structure object
        A  (magnetic or not) structure containing impurity in form of Pymatgen 
        Structure object 
    mark_muon :  bool
        Specify the interstitial impurity properties. Default=True
    moment_to_polarization : bool
        If True to scale moment to maximum spin polarizations. Default=True
    
    Returns : tuple
    -------
    pymatgen_structure : Structure object
        A pymatgen structure with modified type of species/elements
    has_spin : int
        spin type that represent the magnetic ordering type
    magnetic_elements_kind : dict
        A dictionary of type magnetic ions and their values
    """
    # analyze magnetism
    magnetic_structure = CollinearMagneticStructureAnalyzer(pymatgen_structure, 
                                                            make_primitive=False)

    has_spin = 0
    if magnetic_structure.is_magnetic:
        if magnetic_structure.is_collinear:
            has_spin=2
        else:
            has_spin=4
    else:
        has_spin=1

    # check and force a collinear structure
    if has_spin == 4:
        ncol_to_col = check_and_force_collinear(pymatgen_structure)
        if ncol_to_col == True:
            has_spin = 2
       # dont care if it is noncollinear
       #if  brute_force_to_collinear:
       #    has_spin = 2

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
        kind_values[-1] = 'No'
    # Finally return structre, type of magnetic order
    # and a dictionary for magnetic elements, where all the kinds are reported together
    # with the value of the spin.

    # just to make non magnetic calculations
    #if non_magnetic_calculation:
    #   has_spin = 1; #magnetic_elements_kinds = {}

    return (pymatgen_structure.copy(site_properties={'kind_name': kind_values}), 
            has_spin, magnetic_elements_kinds )


# In[8]:


def run_wc(
    index, 
    name, 
    struct, 
    group, 
    codename, 
    pseudo_family, 
    k_distance=0.20, 
    scale_element_init_moment={}, 
    input_namelists={}, 
    mu_plus=False, 
    code_runtime_options=None
):
    """
    This function converts a given initial structure into an AiiDA readable format
    to runs an AiiDA RELAX Workchain.
     
    Parameters
    ----------
    index : int
        An index of the structure supply eg. index=0
    name : str
        The name of the structure or calculations eg. name='Fe'
    struct : Structure object
        The defect structure
    group : str
        The name of the group to store AiiDA nodes on AiiDA database. Its advisable to keep track
        of this name to make sure you organize your data
    codename : str
        The name of the code configured on AiiDA database to perform this type of calculations
    pseudo_family : str
        A pseudopotential family configured on AiiDA database
    k_distance : float
        The density of k-point mesh use for k-point integration in DFT. For more see this
        material cloud page for its usage: https://www.materialscloud.org/work/tools/qeinputgenerator
        Default=0.2.
    scale_element_init_moment : dict
        A user define dictionary type of magnetic ions to scale its magnetic moment.
        Default={}.
    input_namelists : dict
        A user define Quantum ESPRESSO (QE) input namelists. Default={}
    mu_plus : bool
        If True, to specify a total charge of the system and initialise a
        starting charge of muon. Defualt=False
    code_runtime_options : None
        The HPC requirements of number of nodes and k-points. Depends on the code 
        configure in `codename`.
        
    Returns
    -------
    Submit calculation to AiiDA daemon 
    """

    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options

    builder = WorkflowFactory('quantumespresso.pw.relax').get_builder()
    builder.relaxation_scheme = Str('relax')

    builder.metadata.label = "{}-{}".format(name, index)
    builder.metadata.description = "Muon site relaxation workchain for {}, with initial position {} {} {}".format(name, *struct[-1].coords.tolist())
    #builder.meta_convergence=Bool(False)

    StructureData     = DataFactory("structure")
    #builder.structure = StructureData(pymatgen=struct)
    #builder.structure, nspin, magnetic_elements_kinds = pymatgen_to_aiida(struct, StructureData)
    labeled_structure, nspin, magnetic_elements_kinds = analyze_pymatgen_structure(struct)
    builder.structure = StructureData(pymatgen=labeled_structure)
    
    # Append all type of species, magnetic or not
    species = [str(site.specie.symbol) for site in labeled_structure]
    distinct_species =  list(set(species))
    for mag_kinds in magnetic_elements_kinds.values():
        distinct_species.append(list(mag_kinds.keys()))

    if nspin == 2:
        # Set all polarizations to 0.4
        for k in magnetic_elements_kinds.keys():
            for idx in magnetic_elements_kinds[k].keys():
                magnetic_elements_kinds[k][idx] = magnetic_elements_kinds[k][idx] * scale_element_init_moment.get(k, 1.)
    elif nspin == 4:
        raise NotImplemented("Non collinear case not implemented.")

    # Default QE inputs
    Dict = DataFactory('dict')
    parameters_dict = {
        'CONTROL': {
            'calculation': 'relax',
            'restart_mode': 'from_scratch',
            'nstep': 150
      #      'forc_conv_thr': 1,
      #      'etot_conv_thr': 1000
        },
        'SYSTEM': {
            'ecutwfc': 60.,
            'ecutrho': 600.,
            'occupations':'smearing',
            'smearing': 'gaussian',
            'degauss' : 0.02,
            'nspin': nspin, 
        },
        'ELECTRONS': {
            'electron_maxstep': 500,
            'conv_thr'    : 1.0e-7,
            'mixing_beta' : 0.30,
            'mixing_mode' : 'local-TF'
        }
    }
    if nspin == 2:
        parameters_dict['SYSTEM']['starting_magnetization'] = merge_dict_of_dicts(magnetic_elements_kinds)

    if mu_plus:
        parameters_dict['SYSTEM']['starting_charge'] = {'No': 0.6}
        parameters_dict['SYSTEM']['tot_charge'] = 1.

    input_namelistss = copy.deepcopy(input_namelists)
    if input_namelistss:
        # Remove some stuff from input namelist
        # lets comment the stuffs below
        # In my opinion are a mess
        # dictionary changes size for each iterations
        input_namelistss.pop('CONTROL',0)
        input_namelistss.pop('K_POINTS',0)
        input_namelistss.pop('CELL_PARAMETERS',0)
        input_namelistss.pop('ATOMIC_FORCES',0)
        
        if 'SYSTEM' in input_namelistss.keys():
            for kw in ('nat ntyp A B C cosAB cosAC cosBC nbnd tot_charge').split():
                r = input_namelistss['SYSTEM'].pop(kw,'')
                if r:
                    print('Your setting for {} has been removed'.format(r))

            for kw in input_namelistss['SYSTEM'].keys():
                r = ''
                if 'celldm' in kw:
                    r = input_namelistss['SYSTEM'].pop(kw)
                if 'starting_' in kw:
                    r = input_namelistss['SYSTEM'].pop(kw)
                if r:
                    print('Your setting for {} has been removed'.format(r))
        
        # add hubbard parameters
        lda_plus_u_kind = 0
        if 'Hubbard_U' in input_namelistss['SYSTEM'].keys():
            # Get hubbard values from user defined input nameliest
            hubbard_dict = input_namelistss['SYSTEM']['Hubbard_U']
            # The eligible hubbard elements define by user
            hubbard_species = _distinct_species(distinct_species)
            # Specify the hubbard parameters
            hubbard_parameters = set_hubbard_u_parameters(hubbard_species, hubbard_dict)  
            input_namelistss['SYSTEM']['Hubbard_U'] = hubbard_parameters            
            if 'Hubbard_J' in input_namelistss['SYSTEM'].keys():
                # Assumed same as Hubbard U parameters
                input_namelistss['SYSTEM']['Hubbard_J'] = hubbard_parameters
                lda_plus_u_kind = 1
            if 'Hubbard_V' in input_namelistss['SYSTEM'].keys():
                # Assumed same as Hubbard U parameters
                input_namelistss['SYSTEM']['Hubbard_V'] = hubbard_parameters               
                lda_plus_u_kind = 2            
        input_namelistss['SYSTEM']['lda_plus_u_kind'] = lda_plus_u_kind
        parameters_dict = merge(input_namelistss, parameters_dict)

    parameters = Dict(dict=parameters_dict)
 
    builder.base.pseudo_family = Str(pseudo_family)

    #builder.base.kpoints_distance = Float(0.4)
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(builder.structure)
    #set kpoint mesh based on density
    kpoints.set_kpoints_mesh_from_density(k_distance, force_parity=False)
    kpoints.store()

    settings_dict={}
    # Number of k-points to determine K-point integration type
    # Gamma or Automatic by Monkhorst-Pack
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
            #default_options = get_default_options(int(exec_options[0]), int(exec_options[1]), True)
            #builder.base.pw.metadata.options = default_options
            builder.base.pw.metadata.options.resources={'num_machines': int(exec_options[0])}
            builder.base.pw.metadata.options.max_wallclock_seconds = int(exec_options[1])
            builder.base.pw.metadata.options.withmpi = True
            #builder.base.pw.metadata.options.mpirun_extra_params='--map-by socket:PE=8 --rank-by core'.split()
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


# In[9]:


def run_calculation(
    name, 
    struct, 
    group, 
    codename, 
    pseudo_family, 
    k_distance=0.20, 
    scale_element_init_moment={}, 
    input_namelists={}, 
    code_runtime_options=None
):
    """
    This function converts a given initial structure into an AiiDA readable format
    to perform a static SCF calculation.
     
    Parameters
    ----------
    name : str
        The name of the structure or calculations eg. name='Fe'
    struct : Structure object
        The defect structure
    group : str
        The name of the group to store AiiDA nodes on AiiDA database. Its advisable to keep track
        of this name to make sure you organize your data
    codename : str
        The name of the code configured on AiiDA database to perform this type of calculations
    pseudo_family : str
        A pseudopotential family configured on AiiDA database
    k_distance : float
        The density of k-point mesh use for k-point integration in DFT. For more see this
        material cloud page for its usage: https://www.materialscloud.org/work/tools/qeinputgenerator
        Default=0.2.
    scale_element_init_moment : dict
        A user define dictionary type of magnetic ions to scale its magnetic moment.
        Default={}.
    input_namelists : dict
        A user define Quantum ESPRESSO (QE) input namelists. Default={}
    mu_plus : bool
        If True, to specify a total charge of the system and initialise a
        starting charge of muon. Defualt=False
    code_runtime_options : None
        The HPC requirements of number of nodes and k-points. Depends on the code 
        configure in `codename`.
        
    Returns
    -------
    Submit calculation to AiiDA daemon 
    """

    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options
    from aiida.orm import Group

    g, just_created = Group.objects.get_or_create(group)

    code = Code.get_from_string(codename)
    builder = code.get_builder()

    builder.metadata.label = "{} - Unitcell".format(name)
    builder.metadata.description = "Magnetic test for - {}".format(name)

    StructureData     = DataFactory("structure")
    labeled_structure, nspin, magnetic_elements_kinds = analyze_pymatgen_structure(struct, mark_muon=False)
    builder.structure = StructureData(pymatgen=labeled_structure)

    structure_copy = StructureData(pymatgen=labeled_structure)

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
            'calculation': 'scf',
            'restart_mode': 'from_scratch',
        },
        'SYSTEM': {
            'ecutwfc': 60.,
            'ecutrho': 600.,
            'occupations':'smearing',
            'smearing': 'm-v',
            'degauss' : 0.02,
            'nspin': nspin, #
        },
        'ELECTRONS': {
            'conv_thr'    : 1.e-7,
            'mixing_beta' : 0.30,
            'mixing_mode' : 'local-TF'
        }
    }

    #print('Old Dictionary\n', parameters_dict)

    #override
    parameters_dict=merge(input_namelists, parameters_dict)

    #print('New Dictionary\n', parameters_dict)

    parameters = Dict(dict=parameters_dict)
    builder.parameters = parameters

    if nspin == 2:
        parameters_dict['SYSTEM']['starting_magnetization'] = merge_dict_of_dicts(magnetic_elements_kinds)

    KpointsData = DataFactory('array.kpoints')
    #kpoints = KpointsData()
    #kpoints.set_kpoints_mesh([2,2,2],offset=(0,0,0))

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


    builder.pseudos = get_pseudos_from_structure(structure_copy, pseudo_family)

    #builder.metadata.options.resources = resources

    #builder.metadata.options.max_wallclock_seconds = 259200  #86400

    builder.parameters = parameters
    builder.kpoints = kpoints

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
        else:
            exec_options = code_runtime_options.split('|')[0].split()
            default_options = get_default_options(int(exec_options[0]), int(exec_options[1]), True)
            builder.metadata.options = default_options
        if code_runtime_options is None or code_runtime_options == '':
            npool = np.min([4, num_k_points])
            settings_dict['cmdline'] = ['-nk', str(npool), '-ndiag', '1']
        else:
            parallel_options = code_runtime_options.split('|')[1]
            settings_dict['cmdline'] = parallel_options.strip().split()

    clean_workdir = False
    final_scf = False
    if clean_workdir:
        builder.clean_workdir = Bool(True)

    if final_scf:
        builder.final_scf = Bool(True)

    calc = submit(builder)
    if not (g is None):
        g.add_nodes(calc)

    print(name +' magnetic test calculation created  with PK = {}'.format(calc.pk))
    return calc


# In[10]:


def run_hyperfine(
    uuid, 
    structure, 
    sc_size, 
    group,  
    pp_codename, 
    projwfc_codename, 
    symprec,
    if_equivalent_sites,
    if_with_distortions,
    code_runtime_options=None
):
    """This function performs a hyperfine calculation for relax structure
    
    Parameter
    ---------
    uuid : int
        AiiDA uuid of relax structure
    sc_size : str
        Supercell size in a, b and c direction eg "3 3 3".
    group : str
        The name of the group to store AiiDA nodes on AiiDA database. Its advisable to keep track
        of this name to make sure you organize your data
    pp_codename : str
        The name of the code configured on AiiDA database to perform this type of calculations. 
        In this case is for postprocessing of spin density.
    projwfc_codename : str
        The name of the code configured on AiiDA database to perform this type of calculations. 
        In this case is for postprocessing of density of states.        
    symprec: float
        Absolute threshold for checking distance.         
    if_equivalent_sites : bool
        If True, to generate structure for muon equivalent sites.
        Default=True
    if_with_distortions : bool
        If True, to generate structure with distortions for each muon equivalent sites.
        Default=True
    code_runtime_options : None
        The HPC requirements of number of nodes and k-points. Depends on the code 
        configure in `codename`.
        
    Returns
    -------
    Submit calculation to AiiDA daemon 
    """
    from hf import PwContactFieldWorkChain
    from aiida.engine import submit, run, launch
    from aiida.orm import Dict, Group, load_node
    from aiida_symmetry_sites import SiteDistortions
    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options

    g, just_created = Group.objects.get_or_create(group)

    #uuid=57958

    #search_equiv = True

    b = PwContactFieldWorkChain.get_builder()
    calc  = load_node(uuid)

    b.parent_folder = calc.outputs.remote_folder

    #b.pw.code = Code.get_from_string(pp_codename)
    b.pp.code = Code.get_from_string(pp_codename)
    b.projwfc.code = Code.get_from_string(projwfc_codename)

    b.pp.metadata.options=get_default_options(1, 86400, True) # (1, 2000, True) or (1, 43200, True)
    b.projwfc.metadata.options=get_default_options(1, 86400, True)
 
    
    num_sites = 1
    b.equiv_sites = Dict(dict=single_structure_dict(uuid=uuid))
    b.equiv_count = Int(num_sites)
    comment = 'single'
    
    if if_equivalent_sites:
        ESD = SiteDistortions(uuid = uuid,
                              structure = structure,
                              sc_size = sc_size,
                              muon_threshold = symprec,
                              if_pristine = False,
                              if_with_distortions = if_with_distortions
                             )
        
        if_add = True
        structures = ESD.get_positions_as_dict()
        num_sites = len(structures)
        if num_sites==0:
            if_add = False
        if if_add:
            b.equiv_sites = Dict(dict=structures)
            b.equiv_count = Int(num_sites)
            comment = 'multiple'

    print("\n We found : {} ({}) structures".format(num_sites, comment))
    if yes_or_no('Can we proceed?'):
        hf  = submit(b)
        g.add_nodes(hf)


# In[ ]:




