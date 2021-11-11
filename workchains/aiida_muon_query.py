#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import numpy as np
from aiida import orm
from tabulate import tabulate
from itertools import groupby
from aiida import load_profile

load_profile()


# In[2]:


class QueryCalculationsFromGroupError(Exception):
    """
    """
    pass
try:
    import numpy as np
    from aiida import orm
except ImportError:
    raise QueryCalculationsFromGroupError("AiiDA core orm not installed!")
    

class QueryCalculationsFromGroup(object):
    """This class represents a functions Query all types of
    AiiDA UUID (PK) from data group inside AiiDA group.
    """
    
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
    def list_of_strings(lists):
        """This function list of integer strings to lists
        
        Parameters
        ----------
        lists :
              list of values
        Returns
        -------
        lists
        """
        ss = []
        for i, s in enumerate(lists):
            if isinstance(s, str):
                s_ = [int(j) for j in s.split()]
            else:
                s_ = [s]
            ss.append(s_)
        return [item for sublist in ss for item in sublist]  
        
    @staticmethod
    def _exit_status(uuid):
        """Return exit status of a calculation
        
        Parameters
        ----------
        uuid  : int
                PK of a calculation
        Returns
        -------
        exit_status
            exit status of a calculation
        """
        calc  = orm.load_node(uuid)
        exit_status = calc.attributes.get('exit_status')        
        return exit_status
    
    @staticmethod
    def _calc_status(uuid):
        """Return exit status of a calculation
        
        Parameters
        ----------
        uuid  : int
                PK of a calculation
        Returns
        -------
        bool, exit_status
            boolean and exit status of a calculation
        """
        calc  = orm.load_node(uuid)
        exit_status = calc.attributes.get('exit_status') 
        if exit_status == 0:
            return "T", 0
        return "F", exit_status
    
    @staticmethod
    def _energy(uuid):
        """To get the final energy (in Electron Volt) of relax structure
        Parameters:
        -----------
        uuid : int
        """
        calc = orm.load_node(uuid)
        #calc.outputs.output_parameters.get_dict()["energy"]  ##calc.res    # (in eV) # QE-GPU v6.4 parser problem for energy
        return calc.outputs.output_trajectory.get_array("energy")[-1]   # in eV
    
    @staticmethod
    def link_contact(uuid):
        """To check if contact field link is present
        
        Parameters
        ----------
        uuid  : int
                PK of a calculation
        Returns
        -------
        bool        
        """
        calc = orm.load_node(uuid)
        links = calc.get_outgoing().all_link_labels()
        if 'contact_field' in links:
            return True
        return False
    
    @staticmethod
    def link_contact_equiv(uuid):
        """To check if the link of contact field of equivalent sites present
        
        Parameters
        ----------
        uuid  : int
                PK of a calculation
        Returns
        -------
        bool        
        """
        calc = orm.load_node(uuid)
        links = calc.get_outgoing().all_link_labels()
        if 'contact_field_dict' in links and 'contact_field' in links:
            return True
        return False
    
    @staticmethod
    def get_relax_structure(uuid):
        """Get relaxed structure of a calculations in PyMatgen format
        
        Parameters
        -----------
        uuid : int
               PK of a calculation
        Returns
        -------
        Structure
            pymatgen Structure Object
        """
        calc = orm.load_node(uuid)
        return calc.outputs.output_structure.get_pymatgen_structure()  
    
        
    def __init__(
        self, 
        group_uuid = None, 
        process_label = None, 
        process_status_name = 'finished',
        exit_status = 0
    ):
        
        """
        Parameters:
        -----------
        group_uuid : str or int
                     The group uuid or name
        process_label : str
                        The workchain name use as a filter for certain calculations
                        e.g "PwContactFieldWorkChain" for hyperfine field nodes,
                        "PwRelaxWorkChain" for structural relaxation node
                        Default = None, to return all nodes in the group        
        process_status_name : str
                              name of the process type  i.e 'killed', 'excepted' or 'finished'. 
                              Default = 'finished'
        exit_status : int
                      exit status of a calculation, Default = 0
        """

        self.group_uuid = group_uuid
        self.process_label = process_label   
        self.process_status_name = process_status_name.lower()
        self.exit_status = exit_status
        #print('check')
        filters = {}
        if self.process_label is not None or self.process_label != "":
            filters['attributes.process_label'] = {'==': self.process_label}
            if self.process_status_name == 'finished':
                filters['attributes.process_state'] = {'==': 'finished'}
                filters['attributes.exit_status'] = {'==': 0}
            if self.process_status_name == 'excepted': 
                filters['attributes.process_state'] = {'in': ['finished', 'excepted']}
                filters['attributes.exit_status'] = {'>': 0}
            if self.process_status_name == 'all':
                filters['attributes.process_state'] = {'in': ['finished', 'excepted', 'killed']}
                filters['attributes.exit_status'] = {'>=': 0}
        elif self.process_label is None or self.process_label == "":
              filters['attributes.process_state'] = {'in': ['finished', 'excepted']}
              filters['attributes.exit_status'] = {'>=': 0}
        else:    
             raise ValueError('Wrong process state : {}, either "finished", "excepted" or "all" is required!'.format(process_status_name))
        
        self.filters = filters      
                
    def query_nodes(
        self,
        uuid = None,
        proc_label = None,
        proc_status = 'finished',
        exit_status = 0
    ):
        """This function performs the querying 
        Returns:
        --------
               nodes: list
                      the list of all uuid of each calculations within the group
        """
        
        # Lets load the group PK and it is Label(name)
        # And AiiDA QueryBuilder()
        # Handle invalid and new inputs.
        if uuid is None and self.group_uuid is None:
            raise QueryCalculationsFromGroupError("Invalid input parameters")
        elif uuid is not None:
            self.__init__(
                group_uuid=uuid, 
                process_label=proc_label,
                process_status_name=proc_status,
                exit_status=exit_status
            )
  
        filters = self.filters
        group_uuid = self.group_uuid  
        group = orm.load_group(group_uuid)
        group_name = group.label
        qb = orm.QueryBuilder()
        qb.append(orm.Node, tag="nodes", project=["id"], filters=filters)
        qb.append(orm.Group, tag="group", with_node="nodes",  filters={"label":group_name})

        n_nodes = qb.distinct().all()
        res_nodes = [x for x in qb.all()]
        self.nodes = list()
        for i in range(len(n_nodes)):
            self.nodes.append(qb.dict()[i]["nodes"]["id"])
        self.nodes = [i for i in self.nodes if i] 
        return self.nodes
        
    def tabulate_query_nodes(self):
        """tabulate nodes
        """        
        PW = []
        index_ = []
        nodes_ = []
        uuid_ = []
        status_bool_PW = []
        status_number_PW = []
        status_number_nodes = []        
        for i, uuid in enumerate(self.query_nodes()):
            index_.append(i+1)
            nodes_.append(uuid)
            #uuid_.append(orm.load_node(uuid))
            ID = self.find_relaxed_PK(uuid)
            status_bool_PW.append(self._calc_status(ID)[0])
            status_number_PW.append(self._calc_status(ID)[1])
            status_number_nodes.append(self._calc_status(uuid)[1])
            PW.append(ID)
    
        #headers=["##", "Class pk", "Class uuid", "Class error #", "PW pk", "PW error #"]
        headers=["##", "CLASS PK", "CLASS ERROR ##", "PW PK", "PW ERROR ##"]
        data = zip(index_,
                   nodes_,
                   #uuid_,
                   status_number_nodes,
                   PW,
                   status_number_PW                  
                  )
        print(tabulate(data, headers=headers, tablefmt="github")) 

    def find_relaxed_PK(self, uuid):
        """Find relaxed structure UUID for a particular process label
        
        Parameters
        ----------
        uuid : int
               PK of a calculation
        Returns
        -------
        r_node : int
            relaxed structure PK
        """
        calc  = orm.load_node(uuid)
        if self.process_label == "PwRelaxWorkChain":
            o_struct = calc.inputs.structure
        if self.process_label == "PwContactFieldWorkChain":
            o_struct = calc.inputs.parent_folder.creator.inputs.structure
        all_nodes = o_struct.get_outgoing().all_nodes()
        for node in all_nodes:
            if node.process_label == "PwCalculation":
                r_node = node.pk
        return r_node
    
    def _get_all_relaxed(self):
        """To get all the relaxed calculations Pk from the group
        """
        return [self.find_relaxed_PK(uuid) for uuid in self.query_nodes()]
        
    def get_all_relaxed_with_exit_status(self, status=None):
        """Find all relaxed structure with a given exit status calculations
        Parameters:
        ----------
        status :
        """                                          
        if status is None:
            exit_status = [0]
        elif status is not None:
            if isinstance(status, int):
                exit_status = [status]
            elif isinstance(status, str):
                exit_status = [status]               
            elif isinstance(status, list):
                exit_status = list_of_strings(status)
            else:
                raise QueryCalculationsFromGroupError("Invalid status given")        
        nodes = []
        for i, uuid in enumerate(self._get_all_relaxed()):
            if self._exit_status(uuid) in exit_status:
                nodes.append(uuid)
        return nodes        
    
    def get_pw_nodes(
        self, 
        status=None, 
        energy_threshold=1000
    ):
        """Get all relaxed PK sorted and energy less than certain value
        
        Parameters
        -----------
        energy_threshold : float
                           energy threshold in electron volt
        Returns
        -------
        uuid_
            list of nodes
        """
#         if energy_threshold is None:
#             energy_thresh = 1000
#         elif energy_diff is not None:
#             energy_thresh = energy_threshold            
        nodes = self.get_all_relaxed_with_exit_status(status=status)
        energy = []
        for uuid in nodes:
            energy.append(self._energy(uuid))
        uuid_ = []
        sort = sorted(zip(nodes, energy), key = lambda x:x[-1])
        energy = []
        nodes = []
        for n, e in sort:
            nodes.append(n)   
            energy.append(e)
        energy_diff = [abs(en-energy[0]) for en in energy]
        for uuid, edif in zip(nodes, energy_diff):
            if edif <= energy_threshold:
                uuid_.append(uuid)
        return uuid_                
    
    def get_relaxed_nodes(self):
        """To get all uuid
        """
        return self.get_pw_nodes(status=0)
            
    def get_energies(self):
        """To get all energies
        """
        return [self._energy(pk) for pk in self.get_relaxed_nodes()]
        
    def get_positions(self):
        """To get all positions
        """
        positions = []
        for i, uuid in enumerate(self.get_relaxed_nodes()):
            structure = self.get_relax_structure(uuid=uuid)
            positions.append(structure.frac_coords)
        return positions
    
    def get_hyperfine_nodes(self):
        """A collection of 'PwContactFieldWorkChain' query result, 
        sort the PK based on energy.
        
        Returns
        -------
        list
            list of contact hyperfine field nodes
        """

        nodes = self.query_nodes()
        pw1 = []
        pw2 = []
        n_nodes = []
        exit_status_id = [None, 0, 11, 401]
        for uuid in nodes:
            if self._exit_status(uuid) in exit_status_id:
                pw = self.find_relaxed_PK(uuid)
                if self._exit_status(pw) !=0:
                    continue        
                n_nodes.append(uuid)
                pw1.append(pw)
        
        group_pw = group(pw1)
        sublists = [[] for i in group_pw.items()]
        for i, item in enumerate(group_pw.items()):
            pw, index = item[0], item[1]
            pw2.append(pw)
            for j in index:
                sublists[i].append(n_nodes[j])
        nodes1 = []
        for val in sublists:                
            nodes1.append(self._find_complete_contact_nodes(nodes=val))            
        energy = []
        for uuid in pw2:
            energy.append(self._energy(uuid)) 
            
        zipper = zip(nodes1, energy)
        uuid_ = []
        sort = sorted(zipper, key = lambda x:x[-1]) 
        for n, e in sort:
            uuid_.append(n)
        return uuid_
    
    def _find_complete_contact_nodes(self, nodes):
        """
        A collection of 'PwContactFieldWorkChain' query result.
        We choose the best nodes from the list of hyperfine calculations
        that shared thesame relaxed structure. We choose the one that we 
        perform hyperfine calculations for equivalent sites, else we choose anyone
        Parameters:
        -----------
        nodes : list
                list of contact fields calculations              
        Return:
                (int) PK of  hyperfine calculations
        """
        link_1 = []
        link_2 = []
        found = False
        for i, uuid in enumerate(nodes):           
            if self.link_contact_equiv(uuid):
                link_1.append(i)
            if self.link_contact(uuid):
                link_2.append(i)            
        if len(link_1) >= 1:
            return nodes[link_1[0]]
        elif len(link_2) >= 1:
            return nodes[link_2[0]]
        else:
            nodes[0]        
        return nodes[0]


# In[3]:


class QueryNodeEnergyPositions(object):
    """To query Node, Energy and positions
    """
    
    @staticmethod    
    def get_pwcalc_uuid_from_relaxation_workchain(self, uuid):
        """To get pw.x QE calculations node for a given workchain uuid
        Parameters:
        -----------
        uuid : int
               Workchain uuid
        Returns: list
                 the node of relaxed PW calculation               
        """
        calc = orm.load_node(uuid)
        o_struct = calc.inputs.structure
        all_nodes = o_struct.get_outgoing().all_nodes()
        for node in all_nodes:
            if node.process_label == "PwCalculation":
                r_node = node.pk
        return r_node
    
    @staticmethod
    def get_pwcalc_uuid_from_hyperfine_workchain(self, uuid):
        """To get pw.x QE calculations node for a PwContactFieldWorkChain (hyperfine calc.)
        Parameters:
        -----------
        uuid : int
               Workchain uuid
        """
        calc  = orm.load_node(uuid)
        o_struct = calc.inputs.parent_folder.creator.inputs.structure
        all_nodes = o_struct.get_outgoing().all_nodes()
        for node in all_nodes:
            if node.process_label == "PwCalculation":
                r_node = node.pk
        return r_node
    
    @staticmethod
    def is_calc_finished(self, uuid):
        """to check if exit status of the calculation
        Parameters:
        -----------
        uuid  : int
        """
        calc=orm.load_node(uuid)
        if calc.attributes.get("exit_status") == 0:
            return uuid 
    
    @staticmethod
    def get_pw_energy(self, uuid):
        """To get the final energy of PwCalculation final relax structure
        Parameters:
        -----------
        uuid : int
        """
        calc = orm.load_node(uuid)
        #calc.outputs.output_parameters.get_dict()["energy"]  ##calc.res    # (in eV) # QE-GPU v6.4 parser problem for energy
        return calc.outputs.output_trajectory.get_array("energy")[-1]   # in eV
    
    @staticmethod
    def get_relax_positions(self, uuid):
        """To get all the bfgs step relax positions
        Parameters:
        -----------
        uuid : int
        """
        calc = orm.load_node(uuid)
        return calc.outputs.output_trajectory.get_array("positions")
    
    def get_relax_final_positions(self, uuid):
        """To get all the bfgs step relax positions
        Parameters:
        -----------
        uuid : int
        """
        return self.get_relax_positions(uuid)[-1]
    
    @staticmethod
    def get_relax_final_cell_params(self, uuid):
        """To get final celll paramater
        Parameters:
        -----------
        uuid : int
        """
        calc = orm.load_node(uuid)
        return calc.outputs.output_trajectory.get_cells()[-1]
    
    @staticmethod
    def get_species(self, uuid):
        """To get species (element)
        Parameters:
        -----------
        uuid : int
        """
        calc = orm.load_node(uuid)
        species = calc.outputs.output_trajectory.get_array("atomic_species_name")
        species[-1] = "H"
        return species
    
    @staticmethod
    def get_ase_structure(self, uuid):
        """To get ASE structure objects
        Parameters:
        -----------
        uuid : int
        """
        calc = orm.load_node(uuid)
        return calc.outputs.output_structure.get_ase()
    
    
    def __init__(self, nodes, frac_coords = False):
        self.nodes = nodes
        self.frac_coords = frac_coords   
        
        """
        Parameters:
        -----------
        nodes : list
                the list of nodes query from the group
        frac_coords : bool
                      If True, for fractional coordinates. Default=False
        """
    
    def list_finished_pwcalc_uuid(self):
        """Check from list of nodes which calculation finished
        """
        nodes1 = []
        for uuid in self.nodes:
            nodes1.append(self.get_pwcalc_uuid_from_relaxation_workchain(uuid))
        nodes2 = []
        for uuid in nodes1:
            nodes2.append(self.is_calc_finished(uuid))
        nodes = [i for i in nodes2 if i]
        return nodes
    
    def get_uuid(self):
        """To get all uuid
        """
        uuid = [pk for pk in self.list_finished_pwcalc_uuid()]
        return [i for n, i in enumerate(uuid) if i not in uuid[:n]]
            
    def get_energies(self):
        """To get all energies
        """
        energies = [self.get_pw_energy(pk) for pk in self.list_finished_pwcalc_uuid()]
        return energies
        
    def get_positions(self):
        """To get all positions
        """
        cart_coords = []
        frac_coords = []
        cell_params = []
        for pk in self.list_finished_pwcalc_uuid()  :
            cart_coords.append(self.get_relax_final_positions(pk))
            cell_params.append(self.get_relax_final_cell_params(pk))
    
        if self.frac_coords:            
            for pos, cell in zip(cart_coords, cell_params):
                frac_coords.append(np.dot(pos,np.linalg.inv(cell))%1.)
            return frac_coords
        return cart_coords

    def sort_uuid(self):
        """
        """
        uuid_list1 = self.nodes
        uuid_list2 = self.get_uuid()
        energies = self.get_energies()
        zipper = zip(uuid_list1, uuid_list1, energies)
        uuid_ = []
        sort = sorted(zipper, key = lambda x:x[-1]) 
        for (l1, l2, e) in sort:
            uuid_.append(l1)
        return uuid_


# In[4]:


def safe_div(x, y):
    return 0. if abs(float(y)) == 0. else x / y

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
def find_key_and_value(dic, value):
    """
    """
    for key, values in dic.items():
        if isinstance(values, collections.Iterable):
            for val in values:
                if val==value:
                    return key, val
        else:
            if value==values:
                return key, value
    
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
    
def ratio_of_magmom(uuid, structure):
    """To check the ratio of the magnetic moment between the calculated and experimental magnetic moments
    Parameters:
    -----------
               uuid : (int)
                     The uuid identifier of the calculations
               structure : (Structure)
                         Structure object to obtain the experimental moments
    Returns: 
    """    
    from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
    
    calc = orm.load_node(uuid)
    species = calc.outputs.output_trajectory.get_array("atomic_species_name")
    #print(species)
    output_moment = calc.outputs.output_trajectory.get_array('atomic_magnetic_moments')
    output_moment = output_moment.reshape(len(species),)
    #print(output_moment)
    
    # pristine structure
    structure1 = structure.copy()
    magnetic_structure = CollinearMagneticStructureAnalyzer(structure1, 
                                                            make_primitive=False)
    
    magnetic_species_and_magmoms = magnetic_structure.magnetic_species_and_magmoms
    magnetic_species_and_magmoms_values = list(magnetic_species_and_magmoms.values())
    magnetic_species_and_magmoms_values = flatten(magnetic_species_and_magmoms_values)
    max_magnetic_species_and_magmoms_values = max(magnetic_species_and_magmoms_values)
    mag_specie, max_spin = find_key_and_value(magnetic_species_and_magmoms, 
                                              max_magnetic_species_and_magmoms_values
                                             )
    #print(mag_specie)
    magnetic_species = []
    magnetic_species_index = []
    for i, msp in  enumerate(species):
        if mag_specie==split_element_and_number(msp)[1]:
            magnetic_species.append(msp)
            magnetic_species_index.append(i)
    magnetic_species_index = np.array(magnetic_species_index)
    #print(magnetic_species_index)
    magnetic_atoms_magmom = np.array(output_moment[magnetic_species_index])
    ratios = []
    index = []
    for i, m in enumerate(magnetic_atoms_magmom):
        index.append(i+1)
        if abs(float(m)) < 1e-3:
            ratios.append('inf')
        else:
            ratios.append(max_spin/m)
    
    tab = zip(index, magnetic_species, magnetic_atoms_magmom, ratios)
    headers=["##", 
             "MAGNETIC SPECIES", 
             "DFT MOMENTS (muB)", 
             "RATIO (max(EXP="+str(max_spin)+")"+"/DFT)"
            ]
    print(tabulate(tab, headers=headers, tablefmt="github"))

