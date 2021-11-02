#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ase
import sys
import itertools
import numpy as np
from copy import deepcopy
from aiida import load_profile
from aiida.orm import load_node
from matplotlib import pyplot as plt
from ase.spacegroup import Spacegroup
from pymatgen import Lattice, Structure, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations


load_profile()


# In[2]:


def slicer(input_list, length_to_split):
    """Given a list 'input_list' and split it into sublists of lists
    of different length 'length_to_split'
    """
    from itertools import islice
    input_list = iter(input_list)
    return [list(islice(input_list, elem)) for elem in length_to_split] 
    
def computeTicks(x, step = 5):
    """
    https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib
    Computes domain with given step encompassing series x
    @ params
    x    - Required - A list-like object of integers or floats
    step - Optional - Tick frequency
    """
    import math
    xMax, xMin = math.ceil(max(x)), math.floor(min(x))
    dMax, dMin = xMax + abs((xMax % step) - step) + (step if (xMax % step != 0) else 0), xMin - abs((xMin % step))
    return range(dMin, dMax, step)

def set_shared_label(a, xlabel, ylabel, labelpad = 0.01, figleftpad=0.05, fontsize=12):
    """Set a y label shared by multiple axes
    https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
    Parameters
    ----------
    a: list of axes
    ylabel: string
    labelpad: float
        Sets the padding between ticklabels and axis label"""

    f = a[0,0].get_figure()
    f.canvas.draw() #sets f.canvas.renderer needed below

    # get the center position for all plots
    top = a[0,0].get_position().y1
    bottom = a[-1,-1].get_position().y0

    # get the coordinates of the left side of the tick labels
    x0 = 1
    x1 = 1
    for at_row in a:
        at = at_row[0]
        at.set_ylabel('') # just to make sure we don't and up with multiple labels
        bboxes, _ = at.yaxis.get_ticklabel_extents(f.canvas.renderer)
        bboxes = bboxes.inverse_transformed(f.transFigure)
        xt = bboxes.x0
        if xt < x0:
            x0 = xt
            x1 = bboxes.x1
    tick_label_left = x0

    # shrink plot on left to prevent ylabel clipping
    # (x1 - tick_label_left) is the x coordinate of right end of tick label,
    # basically how much padding is needed to fit tick labels in the figure
    # figleftpad is additional padding to fit the ylabel
    plt.subplots_adjust(left=(x1 - tick_label_left) + figleftpad)

    # set position of label, 
    # note that (figleftpad-labelpad) refers to the middle of the ylabel
    a[-1,-1].set_ylabel(ylabel, fontsize=fontsize)
    a[-1,-1].yaxis.set_label_coords(figleftpad-labelpad,(bottom + top)/2, transform=f.transFigure)

    # set xlabel
    y0 = 1
    for at in a[-1]:
        at.set_xlabel('')  # just to make sure we don't and up with multiple labels
        bboxes, _ = at.xaxis.get_ticklabel_extents(f.canvas.renderer)
        bboxes = bboxes.inverse_transformed(f.transFigure)
        yt = bboxes.y0
        if yt < y0:
            y0 = yt
    tick_label_bottom = y0

    a[-1, -1].set_xlabel(xlabel, fontsize=fontsize)
    a[-1, -1].xaxis.set_label_coords((top + bottom) / 2, tick_label_bottom - labelpad, transform=f.transFigure)
    
    


# In[3]:


class SiteDistortionsError(Exception):
    """
    """
    pass
try:
    import numpy as np
except ImportError:
    raise SiteDistortionsError("Invalid inputs!")
    

class SiteDistortions:
    """Equivalent sites structure object. We used a muon site to generate
    Equivalent muon sites structure with nuclei distortions obtain from
    original relaxed structure containing muon site.
    """
    
    #Lazily wrote! We dont mind if it works!
    
    @staticmethod
    def group(items):
        """group a list containing identical items 
        into group  with their index
        """
        from itertools import groupby
        result = {
            key: [item[0] for item in group]
            for key, group in groupby(sorted(enumerate(items), key=lambda x: x[1]), lambda x: x[1])
        }
        return result
    
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
        """Cell parameters of a given structure
        
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
    def _create_spacegroup(dataset):
        """manually create a spacegroup of the choosen data we want. 
        In this case we are makig sure we exploit all the symmetry 
        operation of the cell(or supercell)
        
        Parameters
        ----------
        dataset : tuple
                  space group informations
        Retuns
        ------
        ase spacegroup data
        """
        import ase
        spgnbr    = dataset['number'] 
        hall      = dataset['international'] 
        rot       = dataset['rotations'] 
        trans     = dataset['translations']
        subtrans  = dataset['origin_shift']
        spg = ase.spacegroup.spacegroup.spacegroup_from_data(no=int(spgnbr),
                                                             symbol=hall,
                                                             centrosymmetric=None,
                                                             scaled_primitive_cell=None,
                                                             reciprocal_cell=None,
                                                             subtrans=subtrans,
                                                             sitesym=[],
                                                             rotations=rot,
                                                             translations=trans,
                                                             datafile=None
                                                             )
        return spg
        

        
    def __init__(
        self, 
        uuid,
        structure,              # pristine unit cell
        sc_size = "1 1 1",
        signicant_figures = 4,
        muon_threshold = 1e-2,
        if_pristine=False,
        if_with_distortions=True
#         reduced_sites = False,
#         reduced_distance = 10.0
    ):
        """
        
        Parameters
        -----------
        uuid : int
            The PK for relax structure              
        structure : Structure object
            Pymatgen Structure Object                   
        sc_size : str
            Supercell size in a, b and c direction eg "3 3 3"
        signicant_figures : int
            Round positios to significant figures. Default=4
        muon_threshold : float
            Absolute threshold for checking distance.      
        if_pristine : bool
            If True, to generate symmetry pristine structure for muon equivalent sites. 
            Default=False
        if_with_distortions : bool
            If True, to generate structure with distortions for each muon equivalent sites.
            Default=True
            
        Returns
        -------
        """
        #
        """
#         reduced_sites: Bool
#             If True, (planning for something). 
#             Default=False
#         reduced_distance : float
#             planning for something        
        """
        #
                        
        self.relax_uuid = uuid
        self.structure = structure
        self.sc_size = sc_size
        self.signicant_figures = signicant_figures
        self.muon_threshold = muon_threshold
        self.if_pristine = if_pristine
        self.if_with_distortions = if_with_distortions
        
        # This parameters for future
#         self.reduced_sites = reduced_sites
#         self.reduced_distance = reduced_distance        

        if sc_size is None or sc_size == "":
            self.sc_size = "1 1 1"
        else:
            self.sc_size = sc_size
            
        self.pristine_structure_supercell = self.structure.copy()
        self.pristine_structure_supercell.make_supercell([int(x) for x in self.sc_size.split()])
        
        
        # Check if pritine structure is compatible to supercell size 
        if len(self.pristine_structure_supercell) != self.n_atoms-1:
            raise SiteDistortions('Invalid inputs Structure or supercell size')
        
        SA = SpacegroupAnalyzer(self.pristine_structure_supercell, 
                                symprec=self.muon_threshold
                               )  
        self._SA = SA

        self._SG = SA.get_space_group_operations()
        self._PG = SA.get_point_group_operations()        
        self._SGO = SpacegroupOperations(SA.get_space_group_number(), 
                                         SA.get_space_group_symbol(), 
                                         SA.get_symmetry_operations(cartesian=False))
        

    @property
    def pristine_structure(self):
        """
        """
        return self.structure.copy()      

    @property
    def initial_structure(self):
        """Pymatgen initial structure containing muon
        """
        return self.pymatgen_input_structure(self.relax_uuid)

    @property
    def mu_i(self):
        """Muon index in the structure
        """
        return self.muon_index(self.initial_structure)
 
    def _initial_structure(self):
        """Initial structure
        """
        structure = self.initial_structure.copy()
        mu_site = structure.pop(self.mu_i)
        return structure, mu_site
    
    @property
    def initial_muon_site(self):
        """Trial initial muon position.
        """
        return self._initial_structure()[1]

    @property
    def initial_ionic_positions(self):
        """
        """
        return self._initial_structure()[0]

    def __initial_host_structure(self):
        """
        """
        return self._initial_structure()[0] 
    
    @property
    def initial_host_structure(self):
        """
        """
        return self.__initial_host_structure()
    
    @property
    def relax_structure(self):
        """Pymatgen relax structure containing muon
        """
        return self.pymatgen_output_structure(self.relax_uuid)
    
    @property
    def mu_f(self):
        """Muon index in the structure
        """
        return self.muon_index(self.relax_structure)
    
    def _relax_structure(self):
        """Relax structure
        """
        structure = self.relax_structure.copy()
        mu_site = structure.pop(self.mu_f)
        return structure, mu_site
 
    @property
    def relax_muon_site(self):
        """Relax muon position.
        """
        return self._relax_structure()[1]

    def __relax_host_structure(self):
        """
        """
        return self._relax_structure()[0]
    
    @property
    def relax_host_structure(self):
        """
        """
        return self.__relax_host_structure()
    
    @property
    def species_(self):
        """Ionic species of the structure
        """
        return self._species(self.relax_uuid)
    
    @property
    def get_species(self):
        """
        """
        species = self.species_
        species[self.mu_f] = 'H'
        species = [self.split_element_and_number(s)[-1] for s in species]
        self.species = species
        return species

    @property
    def n_atoms(self):
        """
        """
        return len(self.get_species) 
    
    @property
    def cell_parameter(self):
        """
        """
        return self._cell_parameters(self.relax_uuid)
    
    def species_label_and_index(self):
        """Returns order species and their index, Hydrogen is at the end"""
        species = self.species_
        all_species = self.group(species)
        # make sure 'H' is at the end for easy visibility
        H_mu = all_species.pop('No')
        all_species.setdefault('H', H_mu)
        species_index = list(all_species.values())
        len_each_label = [len(e) for e in species_index]
        species_label = list(all_species.keys())
        return species_label, len_each_label, species_index, len(species)
    
    
#     def get_symmetry_details(self, pymatgen_structure, symprec=1e-3):
#         """
#         """
#         symprec=self.muon_threshold
#         SA = SpacegroupAnalyzer(pymatgen_structure, symprec=symprec)
#         return SA.get_space_group_number(), SA.get_space_group_symbol()
    
#     def get_muon_symmetry_threshold(self):
#         """
#         """
#         n=100        
#         min_muprec = 1e-4
#         max_muprec = np.linalg.norm(self.initial_muon_site.coords-self.relax_muon_site.coords)
#         random_thres = np.sort(np.random.random(n))
#         random_thres = [(min_muprec + val*(max_muprec-min_muprec)) for val in random_thres]
#         orig_muprec = self.get_symmetry_details(self.pristine_structure_supercell)
#         structure = self._relax_structure()[0]
#         for i, thres in enumerate(random_thres):
#             if self.get_symmetry_details(structure, symprec=thres)==orig_muprec:
#                 return thres
            
#     def create_spacegroup(self):
#         """manually create a spacegroup of the choosen data we want. 
#         In this case we are makig sure we exploit all the symmetry 
#         operation of the cell(or supercell)
#         """
#         import ase
#         SA = self._SA
#         dataset = SA.get_symmetry_dataset()
#         spgnbr    = dataset['number'] 
#         hall      = dataset['international'] 
#         rot       = dataset['rotations'] 
#         trans     = dataset['translations']
#         subtrans  = dataset['origin_shift']
#         spg = ase.spacegroup.spacegroup.spacegroup_from_data(no=int(spgnbr),
#                                                              symbol=hall,
#                                                              centrosymmetric=None,
#                                                              scaled_primitive_cell=None,
#                                                              reciprocal_cell=None,
#                                                              subtrans=subtrans,
#                                                              sitesym=[],
#                                                              rotations=rot,
#                                                              translations=trans,
#                                                              datafile=None
#                                                              )
#         return spg

#     def _ase_muon_find_equiv(self, mu_site):
#         """
#         """
#         eps=self.mu_prec
# #         sg = self.create_spacegroup()
#         sg = self._create_spacegroup(self._SA.get_symmetry_dataset())
#         sites = sg.equivalent_sites(mu_site, symprec=eps, onduplicates='warn')[0]
#         return sites

#     def ase_muon_find_equiv(self):
#         """
#         """
#         # remove the first one which we already have
#         mupos = self.relax_muon_site.frac_coords
#         return self._ase_muon_find_equiv(mu_site=mupos)[1:]
    
#     @property
#     def muon_equiv_sites(self):
#         """
#         """
#         return self.ase_muon_find_equiv()#     def get_symmetry_index(self, mu_index, s_mupos):
#         """Return symmetry index that gives exact s_mupos (from equivalent site)
        
#         Parameters
#         ----------
#         mu_index : int
#                    muon index
#         s_mupos : nd.array
#                   muon site
#         Returns
#         tuple
        
#         """
#         operations = self._SG
#         r_mupos = np.around(self.relax_muon_site.frac_coords, int(self.n_prec))
#         s_mupos = np.around(s_mupos, int(self.n_prec))
#         all_index = []        
#         positions = []
#         cart_coords = []        
#         # we are trying to find all the symmetry operations that will
#         # let the choosen equivalent site invariant
#         for i, op in enumerate(operations):
#             if op.are_symmetrically_related(r_mupos, s_mupos, float(self.mu_prec)): #0.099999
#                 newpos = op.operate(r_mupos)
#                 all_index.append(i)
#                 #positions.append(newpos)  
#         return mu_index, all_index    
    
#     def choosen_index(self):
#         """
#         """
#         #indexes = self.choose_closest_equivalent_sites()
#         frac_coords = self.ase_muon_find_equiv()
#         sg_index = []
#         mu_index = []
#         for i, pos in enumerate(frac_coords):
#             mu_ind, sg_ind = self.get_symmetry_index(i, frac_coords[i])
#             if not sg_ind:
#                 continue
#             mu_index.append(mu_ind)
#             sg_index.append(sg_ind[-1])                
#         return mu_index, sg_index
        

#     def get_symmetry_index(self, mu_index, s_mupos):
#         """Return symmetry index that gives exact s_mupos (from equivalent site)
        
#         Parameters
#         ----------
#         mu_index : int
#                    muon index
#         s_mupos : nd.array
#                   muon site
#         Returns
#         tuple
        
#         """
#         operations = self._SG
#         r_mupos = np.around(self.relax_muon_site.frac_coords, int(self.n_prec))
#         s_mupos = np.around(s_mupos, int(self.n_prec))
#         all_index = []        
#         positions = []
#         cart_coords = []        
#         # we are trying to find all the symmetry operations that will
#         # let the choosen equivalent site invariant
#         for i, op in enumerate(operations):
#             if op.are_symmetrically_related(r_mupos, s_mupos, float(self.mu_prec)): #0.099999
#                 newpos = op.operate(r_mupos)
#                 all_index.append(i)
#                 #positions.append(newpos)  
#         return mu_index, all_index    
    
#     def choosen_index(self):
#         """
#         """
#         #indexes = self.choose_closest_equivalent_sites()
#         frac_coords = self.ase_muon_find_equiv()
#         sg_index = []
#         mu_index = []
#         for i, pos in enumerate(frac_coords):
#             mu_ind, sg_ind = self.get_symmetry_index(i, frac_coords[i])
#             if not sg_ind:
#                 continue
#             mu_index.append(mu_ind)
#             sg_index.append(sg_ind[-1])                
#         return mu_index, sg_index

#     def get_positions_with_distortions(self):
#         """Returns a structure of mupos with distortions
#         """
# #         if self.no_distortions:
# #             return self.get_positions_with_no_distortions()
        
#         # pristine and relaxed positions
#         n_prec = self.n_prec
#         pristine_pos = self.initial_host_structure.frac_coords # in frac_coords
#         pristine_pos = np.around(pristine_pos, int(n_prec))
#         relaxed_pos =  self.relax_host_structure.frac_coords
#         relaxed_pos =  np.around(relaxed_pos, int(n_prec))
#         disp1 = relaxed_pos - pristine_pos  # displacement
        
#         SG = self._SG
#         PG = self._PG
#         """
#         # use symmetry index 'index_op' to find the equivalent muon site of 'i_mupos' i.e 'e_mupos'
#         # Apply space group operations to ionic positions and 
#         # point group operations to the displacement (this keeps distance fixed)
#         """
#         species = self.get_species
#         mu_index, sg_index = self.choosen_index()
#         muon_sites  = self.muon_equivalent_sites
#         #what about distinguishing?
#         list_positions = []
#         mui_index = []
        
#         mu_equiv_frac_coords  = self.muon_equivalent_sites
#         for ix, index in enumerate(mu_index):
#             mui = mu_index[ix]    # muon index for clarifications
#             sgi = sg_index[ix]    # spacegroup index
#             # pristince positions with symmetry operations
#             pristine_pos2 = SG[sgi].operate_multi(pristine_pos)                   
#             disp2 = PG[sgi].operate_multi(disp1)
#             positions = []
#             for i, p0 in enumerate(pristine_pos):
#                 for j,p1 in enumerate(pristine_pos2):
#                     if np.all(np.abs(p0%1-p1%1) < float(self.mu_prec)):
#                         positions.append(pristine_pos2[j]%1+disp2[j])
#             positions.append(muon_sites[mui])
#             if len(positions) != len(species):
#                 continue
#             mui_index.append(mui)
#             list_positions.append(positions)        
            
#         return np.array(list_positions), mui_index


    
    def _muon_equivalent_sites(self):
        """
        """
        symprec=self.muon_threshold
        sg = self._create_spacegroup(self._SA.get_symmetry_dataset())
        mupos = np.around(self.relax_muon_site.frac_coords, int(self.signicant_figures))
        sites = sg.equivalent_sites(mupos, symprec=symprec, onduplicates='warn')[0]
        return sites[1:]
    
    @property
    def muon_equivalent_sites(self):
        """
        """
        return np.array(self._muon_equivalent_sites())
    
    @property
    def n_muon_sites(self):
        """
        """
        return len(self.muon_equivalent_sites)
    
    @property
    def periodic_muon_site(self):
        """The periodic single muon site"""
        structure = self.pristine_structure_supercell.copy()
        mupos = self.relax_muon_site.frac_coords
        return PeriodicSite("H", mupos, structure.lattice, coords_are_cartesian=False)

    @property
    def periodic_muon_equivalent_site(self):
        """get muon position as a PeriodicSite
        
        Parameters
        ----------
        positions : nd.array
                    3D muon positions (including muon)
        Returns: 
               3DXN periodic muon sites 
        """
        
        #frac_coords = self.ase_muon_find_equiv()
        positions = self.muon_equivalent_sites
        structure = self.pristine_structure_supercell.copy()
        periodic_sites = list()
        for mu, i in enumerate(positions):
            periodic_sites.append(PeriodicSite("H", [mu_i for mu_i in positions[mu]], 
                                               structure.lattice, coords_are_cartesian=False))
        return periodic_sites   
    
    def generate_positions_with_equal_distortions(self):
        """Here we assume all symmetry equivalent sites share the same distortions
        """
        relax_position = self.relax_host_structure.frac_coords
        if self.if_pristine:
            relax_position = self.pristine_structure_supercell.frac_coords
        muon_sites = self.muon_equivalent_sites
        index = [i+1 for i in range(self.n_muon_sites)]
        positions = []
        for mui, mupos in enumerate(muon_sites):
            positions2 = []
            for j, ionic_pos in enumerate(relax_position):
                positions2.append(ionic_pos)
            positions2.append(mupos)
            positions.append(positions2)
        return np.array(positions), index        
        
    def generate_positions_with_distinct_distortions(self):
        """This function calculates structural distortion for each symmetric muon
        positions.
        
        Returns:
        tuple
        list_positions, muon_index
        """
        SG = self._SG
        PG = self._PG    
        thresh = self.muon_threshold
        n_sig = self.signicant_figures
        mu_sites = self.muon_equivalent_sites
        
        pristine_pos = np.around(self.initial_host_structure.frac_coords, int(n_sig))
        relax_pos =  np.around(self.relax_host_structure.frac_coords, int(n_sig))
        displacement_1 = relax_pos - pristine_pos  # displacement
               
        list_positions = []
        sgroup_index = []
        muon_index = []
        mu_sites = self.muon_equivalent_sites
        #mu_sites = np.array(mu_sites[0:2])
        # Our calculated relax muon site is 
        r1_site = np.around(self.relax_muon_site.frac_coords, int(n_sig))
        
        # Given a ionic displacement (displacement_1) we apply structural distortions 
        # to each structure that will contain symmetry equivalent positions by:
        # A) Identifying Space Group symmetry operations that each sites (mu_sites)
        # B) For each operation found apply it Spacegroup operation (SG) and PointGroup (PG)
        #    to pristine and displacement respectively. Appplying PG to displacement make sure 
        #    the distortions are unchanged
        # C) Compare pristine position one with amd without SG operations
        
        symmetry_indexes = []  
        muon_index = []
        rot_matrix = []
        for mui, mu_pos in enumerate(mu_sites):
            #muon_index.append(mui+1)     # in human readable
            symmetry_indexes1 = []
            muon_index1 = []
            rot_matrix1 = []
            #print('site : ##{}'.format(mui))
            # A) Here, we find all symmety operations that gives mupos site
            for si, sym_op in enumerate(SG):
                if sym_op.are_symmetrically_related(r1_site, mu_pos, float(thresh)): #0.099999
                    # we save the symmetry index here
                    symmetry_indexes1.append(si)
                    muon_index1.append(mui)
                    rot_matrix1.append(sym_op.rotation_matrix)
            #symmetry_indexes1
            #print('symmetry indexes : {}'.format(symmetry_indexes1))
            # B) each operation; in this case we choose only one?
            spgroup = [symmetry_indexes1[-1]] #symmetry_indexes1 #[symmetry_indexes1[-1]]
            for s, sgi in enumerate(spgroup):
                pristine_pos2 = SG[sgi].operate_multi(pristine_pos)
                displacement_2 = PG[sgi].operate_multi(displacement_1)
                positions = []
                for k, p in enumerate(pristine_pos2):
                    positions.append(pristine_pos2[k]%1+displacement_2[k])
                    
                # C) Check pristine positions
                #  This conditions will always fail for certain operations
                #  Comment above to 2 lines of code to apply condition C) and also
                #  Uncomment below 4 lines to make sure the condition is apply perfectly

#                 for i, p0 in enumerate(pristine_pos):
#                     for j, p1 in enumerate(pristine_pos2):                                    
#                         if np.all(np.abs(p0%1-p1%1) < float(thresh)*10):
#                             positions.append(pristine_pos2[j]%1+displacement_2[j])
                
                # Append muon sites            
                positions.append(mu_pos)
                
                # This condition to make sure we have same number of ionic positions
                # as that of number of species. This condition is as a results of
                # applying condition C) where in some cases the number of ionic positions
                # is less than the number of species for certain  SG operations
                if len(positions) != self.n_atoms:
                    #print('NOT EQUAL POSITIONS!')
                    continue
                # append all positions for each mu_site
                #print('positions =', np.array(positions))
                list_positions.append(positions)
                muon_index.append(mui+1)
        #print('len of structures = ', len(np.array(list_positions))) 
        return np.array(list_positions), muon_index

    def _get_positions(self):
        """Returns generated positions
        
        Returns
        -------
        tuple
        """       
        if self.if_with_distortions:
            positions , index = self.generate_positions_with_distinct_distortions()
        else:
            positions , index = self.generate_positions_with_equal_distortions()

        return np.array(positions), index
    
    @property
    def get_positions(self):
        """Returns generated positions
        """ 
        return self._get_positions()[0]
        
    def save_positions_to_vasp_format(self, file_name=None):
        """save each structure in a 'file_name_s(mui).vasp'  POSCAR format"""
        if file_name is None:
            file_name = str(self.relax_uuid)
        append_file = []                
        species_label, len_each_label, species_index , len_species = self.species_label_and_index()
        n = len(species_label)
        combined = [species_label[i]+'('+str(len_each_label[i])+')' for i in range(n)]        
        list_positions, mu_index = self._get_positions()
        for i, index in enumerate(mu_index):
            mui = mu_index[i]
            positions = list_positions[i]
            file_i = file_name+'_s'+str(mui)+'.vasp'
            append_file.append(file_i)
            with open(file_i,'w') as f: 
                f.write('{} \n'.format(''.join(combined)))  
                #f.write('{} \n'.format(''.join(combined)))  
                f.write('{:f} \n'.format(1.0)) 
                for lat in self.relax_host_structure.lattice.matrix:
                    f.write((' {:16.10f}'*3+' \n').format(*tuple(lat)))                     
                f.write(('  {} '*n+' \n').format(*species_label))
                f.write(('  {} '*n+' \n').format(*len_each_label))               
                f.write('{} \n'.format('Direct'))
                for sp_index in species_index:
                    for spi in sp_index:
                        f.write((' {:16.10f}'*3+' \n').format(*tuple(positions[spi])))                         
            print(' \nOutput file save to file : {} ... \n'.format(file_i), end='', flush=True)                                                      
    
    def get_positions_as_dict(self):
        """make dictionary for all structures"""
        Dict = {}        
        list_positions, mu_index = self._get_positions()
        for i, index in enumerate(mu_index):
            mui = mu_index[i]
            positions = list_positions[i]
            Dict['site'+str(mui)] = {}
            Dict['site'+str(mui)]['lattice'] = self.relax_host_structure.lattice.matrix
            Dict['site'+str(mui)]['species'] = self.species_
            Dict['site'+str(mui)]['positions'] = positions
        return Dict
    
    @property
    def summary(self):
        """
        """
        index = self.get_positions()[1]
        print(' \nExpected number of structures = {} \n'.format(self.n_muon_sites+1))
        print(' \nFound    number of structures = {} \n'.format(len(index)+1))

