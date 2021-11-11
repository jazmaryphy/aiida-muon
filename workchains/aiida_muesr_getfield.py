#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import collections
import numpy as np
from numpy import dot
from copy import deepcopy
from numpy.linalg import inv
from muesr.core import Sample
from tabulate import tabulate
from aiida import load_profile
from pymatgen import Structure
from muesr.core.atoms import Atoms
from ase.spacegroup import Spacegroup
from aiida.orm import Group, load_node
from aiida_symmetry_sites import SiteDistortions
from pymatgen.electronic_structure.core import Magmom
from muesr.engines import locfield, find_largest_sphere
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer

load_profile()


# In[2]:


def flatten(x):
    """Flatten an irregularor regular list of lists of same data type.

    https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists

    Parameters
    ----------
    x : lists
        an irreqular list of lists e.g x = [[1, 2, 3], [4, 5], 6] 
        or x = [[1, 2, 3], [4, 5], [6]] 
    Returns
    ------- 
    A list
    """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
class EstimateFieldContributionsError(Exception):
    """
    """
    pass
try:
    from aiida import load_profile
    
    load_profile() 
except ImportError:
    raise EstimateFieldContributionsError("AiiDA profile or other functions not loaded")
    
class EstimateFieldContributions:
    """This class compute contributions of local fields at the 
    muon sites  with/without contact field and with/without distortions
    """
    
    @staticmethod
    def safe_div(x, y):
        return 0. if abs(float(y)) == 0. else x / y

    @staticmethod
    def flatten(x):
        """Flatten an irregularor regular list of lists of same data type.

        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists

        Parameters
        ----------
        x : lists
            an irreqular list of lists e.g x = [[1, 2, 3], [4, 5], 6] 
            or x = [[1, 2, 3], [4, 5], [6]] 
        Returns
        ------- 
        A list
        """
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]

    @staticmethod
    def find_key_and_value(dic, value):
        """Find key and value of an item with a value=value

        Parameters
        ----------
        dic : dict
            a dictionary containing a value
        value : float, int, str
            an item to find
        Returns
        -------
        key, value
        """
        for key, values in dic.items():
            if isinstance(values, collections.Iterable):
                for val in values:
                    if val==value:
                        return key, val
            else:
                if value==values:
                    return key, value    
    
    @staticmethod
    def split_element_and_number(s):
        """This function separates elementt wit it is index.

        Parameters
        ----------
        s : str
            input string e.g 'Fe1'
        Returns
        -------
        tuple
             (element name, element index)
        """
        return (''.join(filter(str.isdigit, s)) or None,
                ''.join(filter(str.isalpha, s)) or None)
    
    @staticmethod
    def pymatgen_output_structure(uuid):
        """AiiDA output structure in pymatgen format
        
        Parameters
        ----------
        uuid : int
               PK of AiiDA calculation
        Returns
        -------
        Structure Object (pymatgen format)
        """
        calc=load_node(uuid)
        return calc.outputs.output_structure.get_pymatgen_structure() 
    
    @staticmethod
    def pymatgen_input_structure(uuid):
        """AiiDA input structure in pymatgen format
        
        Parameters
        ----------
        uuid : int
               PK of AiiDA calculation
        Returns
        -------
        Structure Object (pymatgen format)
        """
        calc=load_node(uuid)
        return calc.inputs.structure.get_pymatgen_structure()
    
    @staticmethod
    def _species(uuid):
        """Atomic species of a given structure
        
        Parameters
        ----------
        uuid : int
               PK of AiiDA calculation
        Returns
        -------
        list
        """
        calc=load_node(uuid)
        return calc.outputs.output_trajectory.get_array("atomic_species_name")
    
    @staticmethod
    def _cell_parameters(uuid):
        """The relax cell parameters of a given structure
        
        Parameters
        ----------
        uuid : int
               PK of AiiDA calculation
        Returns
        -------
        list
        """
        calc=load_node(uuid)
        return calc.outputs.output_trajectory.get_cells()[-1]
    
    @staticmethod
    def _dft_moments(uuid):
        """Atomic species of a given structure
        
        Parameters
        ----------
        uuid : int
               PK of AiiDA calculation
        Returns
        -------
        list
        """
        calc=load_node(uuid)
        return calc.outputs.output_trajectory.get_array('atomic_magnetic_moments')[-1]
    
    @staticmethod
    def muon_index(structure):
        """Find the muon index in a given structure by identify the element label
        
        Parameters
        ----------
        structure: Structure Object
                   pymatgen structure
        Returns
        -------
        mu_index : int
        """
        mu_index = 0
        for i, site in enumerate(structure):
            if site.species_string in ['H', 'H1, ''No', 'X', 'Xx', 'Xx+']:
                mu_index = i
        if mu_index == len(structure)-1:
            mu_index = -1    #if H is at last     
        return mu_index
    
    @staticmethod
    def save_data(data, filename, logger=''):
        """
        """
        with open(filename, 'w') as outfile:
            outfile.write('# {},  no : {}\n'.format(logger, len(data)))  
            np.savetxt(outfile, data, fmt='%-16.8f')



    def __init__(
        self,
        uuid,
        structure,
        sc_size = None,
        uuid_index = 1,
        if_pristine = False,
        if_scale_moment = True,
        if_with_contact = False,        
        if_scale_contact = False,
        if_equivalent_sites = False,
        if_with_distortions = False,
        file_name = None
    ):
        
        """
        Parameters
        -----------
        uuid : int
               AiiDA PK/UUID
        structure : Structure object
            Pymatgen Structure Object                   
        sc_size : str/None
            Supercell size in a, b and c direction eg "3 3 3". Default ="1 1 1"
        uuid_index : int
            Index to differentiate different uuid. Default = 1
        if_pristine : bool
            If True, to generate symmetry pristine structure for muon equivalent sites. 
            Default=False
        if_scale_moment : bool
            if True, to scale magnetic moment close to experimental values. Default=True
        if_with_contact : bool
            If True, to get or compute contact field. Default=False
        if_scale_contact : bool
            If True, to scale contact field by ratio of experimental to calculated 
            magnetic moment. Default=False 
        if_equivalent_sites : bool
            If True, to consider symmetry equivalent sites. Default=False
        if_with_distortions : bool
            If True, to generate structure with distortions for each muon equivalent sites.
            Default=True
        file_name : None
            file name to save data     

        Returns
        -------
        """
        
        self.uuid = uuid
        self.structure = structure
        self.sc_size = sc_size
        self.uuid_index = uuid_index
        self.if_pristine = if_pristine
        self.if_scale_moment = if_scale_moment
        self.if_with_contact = if_with_contact        
        self.if_scale_contact = if_scale_contact
        self.if_equivalent_sites = if_equivalent_sites
        self.if_with_distortions = if_with_distortions
        self.file_name = file_name
        
        
        # define field data
        self._total_field = []
        self._dipolar_field = []
        self._lorentz_field = []
        self._contact_field_ = []
        
          
        """
        # Try to identify the type of UUID given. 
        # Only two types contains the relaxed structure which can be differentiated by 
        # 1. UUID of 'PwContactFieldWorkChain', or 
        # 2. UUID of 'PwCalculation' label
        # Here, we use AiiDA processs label to differentiate them.
        """
        
        which_calc = load_node(self.uuid)
        exit_status = which_calc.attributes.get("exit_status") 
        exit_status_id = [None, 0, 11, 401]
        if which_calc.process_label == 'PwContactFieldWorkChain':
            """ if no contact field calculation was performed at all perhaps due to error
            messages
            """
            if exit_status in exit_status_id:
                self.load_hyperfine = load_node(self.uuid)
                self.relax_uuid = self._relax_uuid
                self.link_contact = self._link_contact
                self.link_contact_equiv = self._link_contact_equiv
            else:
                self.relax_uuid = self.uuid
                self.link_contact= False
                self.link_contact_equiv = False
                
        elif which_calc.process_label == 'PwCalculation':
            self.relax_uuid = self.uuid
            self.link_contact = False
            self.link_contact_equiv = False
            
        else:
            # you are not giving correct uuid
            raise EstimateFieldContributionsError('Invalid {uuid} input.')
        
        self.load_relax = load_node(self.relax_uuid)
        
        if self.if_with_contact:
            self.link_contact = self.link_contact
            self.link_contact_equiv = self.link_contact_equiv  
        else:
            self.if_scale_contact = False
            self.link_contact = False
            self.link_contact_equiv = False            
        
        
        if sc_size is None or sc_size == "":
            self.sc_size = "1 1 1"
        else:
            self.sc_size = sc_size
            
        self.pristine_structure_supercell = self.structure.copy()
        self.pristine_structure_supercell.make_supercell([int(x) for x in self.sc_size.split()])

        if self.if_scale_contact or self.if_scale_moment:
            self.labels = ['$B_{\mu}^{s}$' , '$B_{D}^{s}$', '$B_{L}^{s}$', '$B_{C}^{s}$']
            self.scale_logger = 'scale'
        else:
            self.labels = ['$B_{\mu}$' , '$B_{D}$', '$B_{L}$', '$B_{C}$']
            self.scale_logger = 'not scale'
    
    @property
    def _relax_uuid(self):
        """Get UUID pk of the relaxed structure
        """
        o_struct = self.load_hyperfine.inputs.parent_folder.creator.inputs.structure
        all_nodes = o_struct.get_outgoing().all_nodes()
        node = [node.pk for node in all_nodes if "PwCalculation" in node.process_label]
        # if more than 1 use the last one
        return node[-1]
    
    @property
    def pristine_structure(self):
        """Copy of pristine (unitcell) structure.
        """
        return self.structure.copy()   
    
    @property
    def equivalent_sites_and_distortions(self):
        """Import SiteDistortions class. 
        #I believe this should be inherited as a parent class
        """
        return SiteDistortions(uuid = self.relax_uuid,
                               structure = self.structure,
                               sc_size = self.sc_size,
                               if_pristine = self.if_pristine,
                               if_with_distortions = self.if_with_distortions
                              ) 

    
    @property
    def initial_structure(self):
        """AiiDA pymatgen initial structure containing muon
        """
        return self.pymatgen_input_structure(self.relax_uuid)

    @property
    def mu_i(self):
        """Muon index in the structure
        """
        return self.muon_index(self.initial_structure)
 
    def _initial_structure(self):
        """Seperate initial structure into initial host and muon positions
        """
        structure = self.initial_structure.copy()
        mu_site = structure.pop(self.mu_i)
        return structure, mu_site
    
    @property
    def initial_muon_site(self):
        """Trial initial muon position.
        
        Returns
        -------
        pymatgen PeriodicSite
        """
        return self._initial_structure()[1]

    @property
    def initial_ionic_positions(self):
        """Initial ionic positions
        
        Returns
        -------
        pymatgen PeriodicSite        
        """
        return self._initial_structure()[0]

    def __initial_host_structure(self):
        """Initial ionic positions
        
        Returns
        -------
        pymatgen PeriodicSite        
        """
        return self._initial_structure()[0] 
    
    @property
    def initial_host_structure(self):
        """Initial host ionic positions
        
        Returns
        -------
        pymatgen PeriodicSite 
        """
        return self.__initial_host_structure()
    
    @property
    def relax_structure(self):
        """AiiDA pymatgen relax structure containing muon
        """
        return self.pymatgen_output_structure(self.relax_uuid)
    
    @property
    def mu_f(self):
        """Muon index in the structure
        """
        return self.muon_index(self.relax_structure)
    
    def _relax_structure(self):
        """Seperate relax structure into relax host and muon positions
        """
        structure = self.relax_structure.copy()
        mu_site = structure.pop(self.mu_f)
        return structure, mu_site
 
    @property
    def relax_muon_site(self):
        """Relax muon position.
        
        Returns
        -------
        pymatgen PeriodicSite        
        """
        return self._relax_structure()[1]

    def __relax_host_structure(self):
        """Relax ionic position.
        
        Returns
        -------
        pymatgen PeriodicSite
        """
        return self._relax_structure()[0]
    
    @property
    def relax_host_structure(self):
        """Relax ionic positions
        
        Returns
        -------
        pymatgen PeriodicSite        
        """
        return self.__relax_host_structure()
    
    @property
    def species_(self):
        """Ionic species of the calculated structure
        """
        return self._species(self.relax_uuid)
    
    @property
    def get_species(self):
        """Periodic table symbols of ionic species
        """
        species = self.species_
        species[self.mu_f] = 'H'
        species = [self.split_element_and_number(s)[-1] for s in species]
        self.species = species
        return species

    @property
    def n_atoms(self):
        """The number of host atoms + muon
        """
        return len(self.get_species) 
    
    @property
    def cell_parameter(self):
        """The matrix of cell parameters
        """
        return self._cell_parameters(self.relax_uuid)
    
    @property
    def pristine_position_with_muon(self):
        """Pristine position of host atoms including muon
        """
        structure = self.pristine_structure_supercell.copy()
        scaled_positions = [site.frac_coords for site in structure]
        scaled_positions.append(self.relax_muon_site.frac_coords)
        return np.array(scaled_positions)

    @property
    def _link_contact(self):
        """Check the availability of hyperfine contact field calculations
        """
        links = self.load_hyperfine.get_outgoing().all_link_labels()
        if 'contact_field' in links:
            return True
        return False       

    @property
    def _link_contact_equiv(self):
        """Check the availability of hyperfine contact field calculations for equivalent sites
        """
        links = self.load_hyperfine.get_outgoing().all_link_labels()
        if 'contact_field_dict' in links and 'contact_field' in links:
            return True
        return False
    
    def _pristine_magnetic_moments_and_direction(self):
        """Magnetic moments of and collinear axis direction
        and separates it into the directions of each moments and 
        collinear spin axis.
        
        Returns
        -------
        moments : numpy.ndarray
        direction : numpy.ndarray
        """
        structure = self.pristine_structure_supercell
        magmoms = structure.site_properties['magmom']
        moments, direction = Magmom.get_consistent_set_and_saxis(magmoms)
        # append for muon
        if self.mu_f==-1:
            moments.append(np.array([0.,0.,0.]))
        else:
            moments.insert(self_mu_f, np.array([0.,0.,0.]))
        if len(moments) != self.n_atoms:
            # to check if calculated structure is compatible to supercell size 
            raise EstimateFieldContributionsError('Invalid inputs in Structure magmoms')
        return moments, direction
    
    def _spin_direction(self):
        """Spin collinear axis direction
        """
        return self._pristine_magnetic_moments_and_direction()[1]
    
    @property
    def spin_direction(self):
        """Spin collinear axis direction
        """
        return self._spin_direction()
           
    def _input_magnetic_moments_vectors(self):
        """ Inputs magnetic moments.
        """
        return self._pristine_magnetic_moments_and_direction()[0]
    
    @property
    def input_magnetic_moments_vectors(self):
        """ Inputs magnetic moments.
        """
        return self._input_magnetic_moments_vectors()
    
    def _input_magnetic_moments(self):
        """Absolute value of input structure magnetic moments
        """
        return np.linalg.norm(self.input_magnetic_moments_vectors, axis=1)
    
    @property
    def input_magnetic_moments(self):
        """Absolute value of input structure magnetic moments
        """
        return self._input_magnetic_moments()

    def _relax_magnetic_moments(self):
        """Calculated magnetic moments
        """
        moments = self._dft_moments(self.relax_uuid) 
        # sets muon magnetic moment to zero
        moments[self.mu_f] = 0 
        return moments
    
    @property
    def dft_magnetic_moments(self):
        """Calculated magnetic moments from AiiDA DFT
        """
        return self._relax_magnetic_moments()

    @property
    def scale_factor(self):
        """Calculate ratio of maximum experimental and calculated magnetic moments and use it as a 
        scaling factor to scale the local magnetic magnetic and contact field to possible
        close to experimental values. 
        """
        species = self.species_
        output_moment = self.dft_magnetic_moments
        output_moment = output_moment.reshape(len(species),)

        # pristine structure
        structure1 = self.structure.copy()
        magnetic_structure = CollinearMagneticStructureAnalyzer(structure1, 
                                                                make_primitive=False)

        magnetic_species_and_magmoms = magnetic_structure.magnetic_species_and_magmoms
        magnetic_species_and_magmoms_values = list(magnetic_species_and_magmoms.values())
        magnetic_species_and_magmoms_values = flatten(magnetic_species_and_magmoms_values)
        max_magnetic_species_and_magmoms_values = max(magnetic_species_and_magmoms_values)
        mag_specie, max_spin = self.find_key_and_value(magnetic_species_and_magmoms, 
                                                       max_magnetic_species_and_magmoms_values
                                                      )
        #print(mag_specie)
        magnetic_species = []
        magnetic_species_index = []
        for i, msp in  enumerate(species):
            if mag_specie == self.split_element_and_number(msp)[1]:
                magnetic_species.append(msp)
                magnetic_species_index.append(i)
        magnetic_species_index = np.array(magnetic_species_index)
        #print(magnetic_species_index)
        magnetic_atoms_magmom = np.array(output_moment[magnetic_species_index])
        max_output_moment = max(magnetic_atoms_magmom)
        # Presumed the calculated moment is never zero
        return max_spin/max_output_moment
    
    def _local_magnetic_moments_vectors(self):
        """ The direction of local magnetic moments of host atoms and muon 
        in Bohr magneton
        """
        local_moments = np.array(self.dft_magnetic_moments)
        moments = ((np.tile(self.spin_direction, [self.n_atoms,1] ).T*local_moments ).T)
        if self.if_scale_moment:
            scale_factor = self.scale_factor
            moments = ((np.tile(self.spin_direction, [self.n_atoms,1]).T*local_moments*scale_factor).T)
        if self.if_pristine:
            moments = np.array(self.input_magnetic_moments_vectors)          
        return moments*(1.+0.j)    
    
    @property
    def get_magnetic_moments(self):
        """Direction of magnetic moment in Bohr magneton
        """
        return self._local_magnetic_moments_vectors()
    
    @property
    def total_magnetization(self):
        """ Total magnetization, the sum of magnetic moments
        """
        return np.sum(self.dft_magnetic_moments)
    
    @property
    def magnetization_per_atoms(self):
        """Magnetization per number of atoms
        """
        return total_magnetization/(self.n_atoms-1)
    
    def _get_contact_field(self):
        """Contact field value in Tesla
        """
        if self.link_contact:
            contact = np.array([self.load_hyperfine.outputs.contact_field.value])
            if self.if_scale_contact:           
                contact *= self.scale_factor 
            return ((np.tile(self.spin_direction, [len(contact),1] ).T*contact).T)            
        return None

    def _get_contact_field_equiv(self):
        """Contact field value of symmetrically equivalent sites in Tesla
        """
        if self.link_contact_equiv:
            contact_dict = self.load_hyperfine.outputs.contact_field_dict.get_dict()
            contact = np.array(list(contact_dict.values()))
            if self.if_scale_contact:
                contact *= self.scale_factor
            return ((np.tile(self.spin_direction, [len(contact),1] ).T*contact).T) 
        return None
    
    def _contact_field(self):
        """ All the contact field
        """
        contact = self._get_contact_field() if self._get_contact_field() is not None else None
        if self.if_equivalent_sites:
            contact = self._get_contact_field_equiv() if self._get_contact_field_equiv() is not None else None
        return contact
    
    @property
    def get_contact(self):
        """Contact field vector
        """
        return self._contact_field()
    
    def _get_frac_position(self):
        """Fractional position of relax structure
        """
        return self.relax_structure.frac_coords%1
    
    def _get_equiv_frac_position(self):
        """Fractional positions of symmetry equivalent structure
        """
        if self.link_contact_equiv:
            return self.load_hyperfine.outputs.structure_positions.get_array('structure_positions')
        return None

    def _get_position(self):
        """All positions
        """
        frac_position = self._get_frac_position()
        # pristine positions
        if self.if_pristine:
            frac_position = self.pristine_position_with_muon        
        if self.if_equivalent_sites:
            # append the Initial muon position
            positions = [frac_position]
            equiv_positions = self.equivalent_sites_and_distortions.get_positions
            if self.link_contact_equiv:
                equiv_positions = self._get_equiv_frac_position()
            for i, pos in enumerate(equiv_positions):
                positions.append(pos)
            return np.array(positions)                         
        return np.array([frac_position])
    
    @property
    def get_position(self):
        """All positions
        """
        return self._get_position()
        
    def _get_muon_sites(self):
        """All muon sites
        """
        muon_sites = [self.relax_muon_site.frac_coords]
        if self.if_equivalent_sites:
            equivalent_sites = list(self.equivalent_sites_and_distortions.muon_equivalent_sites)
            muon_sites.extend(equivalent_sites)                                   
        return np.array(muon_sites)
    
    @property
    def get_muon_sites(self):
        """All muon sites
        """                                   
        return self._get_muon_sites()

    @property
    def n_structures(self):
        """Number of structures
        """
        return len(self.get_muon_sites)
    
    def run_muesr(
        self,
        frac_coords,
        contact_field
    ):
        """ This function use MUESR to calculated local field contributions
        
        Parameters
        ----------
        frac_coords : numpy.ndarray
            fractional coordinates (with muon)
        contact_field : numpy.ndarray
            contact vector in Tesla
            
        Returns
        -------
        total
        dipolar
        lorentz
        contact
        """
        #print('contact field =', contact_field)
        cell = self.cell_parameter
        symbols = self.get_species
        moments = self.get_magnetic_moments
        # Define structure in MuESR
        atoms = Atoms(symbols = symbols, scaled_positions = frac_coords,
                     cell = cell, pbc=True)      
        
        s = Sample()
        s.cell = atoms
        s.new_mm()
        # s.mm.k is already 0
        s.mm.fc = moments
        
        # muon site position in frac_coords
        if self.if_equivalent_sites:
            muon_site = frac_coords[self.mu_f]
        else:
            muon_site = frac_coords[-1]
        # add muon sites
        s.add_muon(muon_site)
        
        if self.if_equivalent_sites:
            # to reduce compuational load. However, the supercell lattice parameters
            # are almost identical
            n=30
            sc = [n,n,n]
        else:
            n=1000
            sc = list(self.relax_structure.lattice.abc)
            sc =[int(np.ceil(n/latt)) for latt in sc] 
            
        r = locfield(s,'s', sc, find_largest_sphere(s,sc), nnn = 2, rcont = 10.0)

        B_d_prime = np.zeros([len(s.muons),3])
        B_d = np.zeros([len(s.muons),3])
        B_l = np.zeros([len(s.muons),3])
        B_c = np.zeros([len(s.muons),3])
        B_t = np.zeros([len(s.muons),3])
        for i in range(len(s.muons)):
            B_d[i] = r[i].D
            B_l[i] = r[i].L            
            B_c[i] = np.array(contact_field)
            B_d_prime[i] = r[i].D + r[i].L   
            B_t[i] = B_d_prime[i] + B_c[i]
        

        return B_t[0], B_d[0], B_l[0], B_c[0], 

    def calculate(self):
        """Run the MUESR calculations for all positions and
        append each contributions to the list
        """
        positions = self.get_position
        contact = self.get_contact
        if contact is None:
            if self.link_contact and self.if_with_contact:
                contact  = [self._get_contact_field()]
                for i in range(self.n_structures-1):
                    contact.append(np.zeros([1,3]))  
            else:
                contact = np.zeros([self.n_structures,3])
            contact = np.array(contact)        
        
        for i, frac_coords in enumerate(positions):
            f=self.run_muesr(frac_coords=frac_coords, contact_field=contact[i])
            self._total_field.append(f[0])
            self._dipolar_field.append(f[1])
            self._lorentz_field.append(f[2])
            self._contact_field_.append(f[3])            
            #print(f)
    
    @property
    def print_logger(self):
        """Logger to show summary of each calculations
        """
        if self.link_contact or self.link_contact_equiv:
            print('    calculation ({}) #{} for a hyperfine with UUID :{} and relax structure:{}'.
                  format(self.scale_logger, 
                         self.uuid_index,
                         self.uuid,
                         self.relax_uuid
                        )
                 )
        else:
            print('    calculation ({}) #{} for a relax structure with UUID :{}'.
                  format(self.scale_logger, 
                         self.uuid_index,
                         self.relax_uuid
                        )
                 )
        
    def total_field(self):
        """Total field vector in tesla
        """
        return np.array(self._total_field)
            
    def dipolar_field(self):
        """Dipolar field vector in tesla
        """
        return np.array(self._dipolar_field)

    def lorentz_field(self):
        """Lorentz field vector in tesla
        """
        return np.array(self._lorentz_field)

    def contact_field(self):
        """Contact field vector in tesla
        """
        return np.array(self._contact_field_)
    
    def magnitude_total_field(self):
        """Magnitude of total field in Tesla
        """
        return np.linalg.norm(self.total_field(), axis=1)
    
    def magnitude_dipolar_field(self):
        """Magnitude of dipolar field in Tesla
        """
        return np.linalg.norm(self.dipolar_field(), axis=1)
    
    def magnitude_lorentz_field(self):
        """Magnitude of lorentz field in Tesla
        """
        return np.linalg.norm(self.lorentz_field(), axis=1)

    def magnitude_contact_field(self):
        """Magnitude of total field in Tesla
        """
        return np.linalg.norm(self.contact_field(), axis=1)

    def get_total_object(self):
        """Generate a data object for total field
        """
        data = []
        for i, f in enumerate(self.magnitude_total_field()):            
            data.append([self.labels[0],    str(self.uuid_index)+'-'+str(i+1),    f])        
        return np.array(data)
    
    def get_dipolar_object(self):
        """Generate a data object for dipolar field
        """
        data = []
        for i, f in enumerate(self.magnitude_dipolar_field()):            
            data.append([self.labels[1],    str(self.uuid_index)+'-'+str(i+1),    f])        
        return np.array(data)
    
    def get_lorentz_object(self):
        """Generate a data object for lorentz field
        """
        data = []
        for i, f in enumerate(self.magnitude_lorentz_field()):            
            data.append([self.labels[2],    str(self.uuid_index)+'-'+str(i+1),    f])        
        return np.array(data)
    
    def get_contact_object(self):
        """Generate a data object for contact field
        """
        data = []
        for i, f in enumerate(self.magnitude_contact_field()):            
            data.append([self.labels[3],    str(self.uuid_index)+'-'+str(i+1),    f])        
        return np.array(data)
    
    def save_muon_sites(self):
        """Store all the muon sites in file
        """
        if self.file_name is None:
            self.file_name = 'muon_sites'        
        filename = self.file_name+'_uuid_'+str(self.relax_uuid)+'_sites_'+str(self.uuid_index)+'.txt'
        print('... Save muon sites in {}'.format(filename))
        self.save_data(self.get_muon_sites, filename, 'Muon site in fractional coordinates of supercell')
        
    def save_total_field(self):
        """Store total field data object in a file
        """
        if self.file_name is None:
            self.file_name = 'total_field'        
        filename = self.file_name+'_uuid_'+str(self.relax_uuid)+'_total_field_'+str(self.uuid_index)+'.txt'
        self.save_data(self.magnitude_total_field(), filename, 'Total Field')

