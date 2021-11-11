#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import numpy as np
from tabulate import tabulate 
from pymatgen import Structure, PeriodicSite, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from aiida_muon_query import QueryCalculationsFromGroup, QueryNodeEnergyPositions


# In[2]:


class GenerateCluster(object):
    """To generate cluster of muon position
    """
    
    @staticmethod
    def group_nodes(index, data):
        """
        """
        return [data[i] for i in index]
    
    @staticmethod
    def group_energies(index, data, ediff=False):
        """
        """
        energies = [data[i] for i in index]
        if ediff:
            energies = [(i-energies[0]) for i in energies]    
        return np.around(energies, 8)
    
    @staticmethod
    def group_positions(index, data, sc_size='1 1 1'):
        """
        """    
        positions = [tuple(data[i].frac_coords%1) for i in index]
        #to unitcell
        positions = np.multiply(positions, [int(x) for x in sc_size.split()])%1
        return np.around(positions, 4)    
    
    
    def __init__(
        self, 
        group_uuid, 
        structure, 
        sc_size, 
        process_label, 
        energy_threshold = 0.01, 
        symprec = 1e-3, 
        if_with_energy = True
    ):
    
        """
        Parameters:
        -----------
        uuid : int 
               A uuid of a given group to generate the cluster
        structure : Structure 
                    A structure object to generate the symmetry operations
        sc_size : int 
                  A list of supercell size to generate symmetry operations
        process_label : str
                        workchain name to query calculation from
        energy_threshold : float 
                     Tolerance of energy difference threshold. Default = 0.01
        symprec : float 
                  Tolerance in atomic distance to test if atoms are symmetrically similar. 
                  Default = 0.1 (if for positions obtain from electronic structure)        
        if_with_energy : bool
                      If False, to not query energy . Default=True                      
        """
        
        self.group_uuid = group_uuid
        self.structure = structure
        self.process_label = process_label    
        self.energy_threshold = energy_threshold
        self.symprec = symprec
        self.if_with_energy = if_with_energy
        #self.frac_coords = frac_coords
        
        if sc_size is None or sc_size == "": 
            self.sc_size = "1 1 1"      
        else:
            self.sc_size = sc_size  
           
        self.structure_sc = structure.copy()
        self.structure_sc.make_supercell([int(x) for x in self.sc_size.split()])  
        
        # Query nodes, energy, positions
        QG = QueryCalculationsFromGroup(self.group_uuid, 
                                        self.process_label
                                       ) 
        

        self.uuid = QG.get_relaxed_nodes()
        self.positions = QG.get_positions()        
        if self.if_with_energy:
            self.energies = QG.get_energies() 
        else:
            self.energies = np.zeros([len(self.uuid)])        
        
        """#To find the space group symmetry operations of the structure 
        """ 
        SA = SpacegroupAnalyzer(self.structure_sc)   
        self.SG = SpacegroupOperations(SA.get_space_group_number(), 
                                       SA.get_space_group_symbol(),
                                       SA.get_symmetry_operations(cartesian=False)
                                      )
             
    
    def periodic_muon_site(self):
        """get muon position as a PeriodicSite
        
        Parameters
        ----------
        positions : nd.array
            numpy.array of fractional position of host atoms (including muon)                    
        Returns: 
        mu_positions : PeriodicSite
            Periodic muon site positions            
        """
        structure = self.structure_sc.copy()
        positions = self.positions
        mu_positions = list()
        for mu_i, mu in enumerate(self.positions):
            mu_positions.append(PeriodicSite("H", [mu_p for mu_p in positions[mu_i][-1]], 
                                             structure.lattice, coords_are_cartesian=False))     
        return mu_positions
        
    def cluster_muon(self): 
        """This function performs the clustering of muon in the foll.
        1. We first cluster the muon w.r.t symmetry inequivalence positions 
        using the muon position index
        2. w.r.t energy we sort within each cluster
        3. We sort all the clusters w.r.t energy
        
        Parameters
        ----------
        nodes : int 
                the list of nodes 
        mu_positions : float 
                       the list of muon positions
        energy : float 
                 the list of DFT total energy 
        sym_operation : list 
                        A list of symmetry operations of the pristine material
        Returns
        -------
        index_sorted2 : list 
                        A list of index of muon position/calculation in each cluster
        """ 
        index0 = []
        index_all = []
        nodes = self.uuid
        energy = self.energies
        SG = self.SG
        mu_positions = self.periodic_muon_site()
        for i , mu_i in enumerate(mu_positions):  
            for j , mu_j in enumerate(mu_positions):  
                if j > i:
                     if j  not in [item for sublist in index_all for item in sublist]:
                         # use PRISTINE symmetry of the cell to check equivalent position 
                         if SG.are_symmetrically_equivalent([mu_positions[i]], [mu_positions[j]], 
                                                                       self.symprec) == True :  #
#                             if abs(energy[i] - energy[j]) > self.energy_threshold:
#                                 print("\n Symmetry Equivalence index i = {} and j = {} with (HIGH) Energy difference = {:.6f} meV"
#                                       .format(i, j, abs(energy[i] - energy[j])*1e3))  
                            index0.append(i) 
                            index0.append(j) 
                            
                            
            index0 = list(set(index0))
            index_all.append(index0)
            index0 = []
            
        index_all=[ind for ind in index_all if ind !=[]] 
        
        #for cases where were only ONE muon position forms a cluster 
        #on its own i.e not symmetry equiv to any
        for i in range(len(mu_positions)):
            if i not in [item for sublist in index_all for item in sublist]: 
                index_all.append([i])  
        
#         print('\n CLUSTER GROUP IN SYMM EQUIV :: {} '.format(index_all))
        
        """Now we group the muon w.r.t position into cluster represented by the 
        muon index (not sorted) and put in index_all.
        Now We sort the index in each cluster w.r.t energy
        """   
            
        index_sorted = []
        new_en = []
        for i in range(len(index_all)):
            for j in index_all[i]:
                new_en.append(energy[j]) 
            index_sorted.append([x for y,x in sorted(zip(new_en, index_all[i]))])
            new_en = []
        
#         print('\n EACH CLUSTER SORTED w.r.t ENERGY :: {} '.format(index_sorted))
        """And now index within the cluster is sorted w.r.t energy and put
        in index_sorted.
        Now we sort index clusters w.r.t. energy
        """
        new_en2 = []
        index1 = []
        index_sorted2 = []
        
        for i in range(len(index_sorted)):
            new_en2.append(energy[index_sorted[i][0]])
            index1.append(i)
            
        index1=[x for y,x in sorted(zip(new_en2, index1))]
        
        for i,j in enumerate(index1):
            index_sorted2.append(index_sorted[j])
        return index_sorted2   
    

    def tabulate_cluster(self):
        """tabulate the cluster
        """
        index_sorted2 = self.cluster_muon()
        
        nodes = self.uuid
        energy = self.energies
        # to meV
        energy = np.array(energy)
        energy *=1000        
        mu_positions = self.periodic_muon_site()
        
        """Now the index clusters is sorted and put in index_sorted2.
        """ 
        #print("\n SORTING CLUSTER w.r.t Energies :: {} ".format(index_sorted2))
        print("\n Number of Cluster Generated :: {} \n".format(len(index_sorted2)))
        
#         flat_index_sorted2 = [item for sublist in index_sorted2 for item in sublist] 
#         all_energy = [energy[i] for i in flat_index_sorted2] 
#         all_energy_diff1  =  [(i-all_energy[0])*1e3 for i in all_energy ]   # in meV
#         all_energy_diff2 = np.zeros(len(flat_index_sorted2)) 
#         for inx, enr in zip(flat_index_sorted2, all_energy_diff1):
#             all_energy_diff2[inx] = enr 
        
        nodes_ = []
        cluster = []
        positions = []
        energy_meV_1 = []
        energy_meV_2 = []
        for i in range(len(index_sorted2)):
            zip0 = zip(index_sorted2[i], 
                       self.group_nodes(index_sorted2[i], nodes), 
                       self.group_positions(index_sorted2[i], mu_positions, self.sc_size), 
                       self.group_energies(index_sorted2[i], energy, ediff=False), 
                       self.group_energies(index_sorted2[i], energy, ediff=True)
                      )           
            
            for (ix, n, p, e1, e2) in zip0:
                cluster.append(i+1)
                nodes_.append(n)
                positions.append(p)
                energy_meV_1.append(e1)
                energy_meV_2.append(e2)
                
        zipped = zip(cluster,
                     nodes_,
                     positions,
#                      energy_meV_1,
                     energy_meV_2
                    )
        
        #headers = ["CLUSTER ##", "PK", "POSITION (x,y,z)", "ENERGY DIFF (meV)", "ENERGY DIFF (meV)"]
        headers = ["CLUSTER ##", "PK", "POSITION (x,y,z)", "ENERGY DIFF/CLUSTER (meV)"]
        print(tabulate(zipped, headers=headers, tablefmt="github"))
        
        
    def distinct_cluster(self, energy_threshold=1.1):
        """Returns symmetry distint sites nodes from the cluster
        """
        index_sorted2 = self.cluster_muon()
        if len(index_sorted2)==0:
            print('NO CALCULATIONS! NO CLUSTER!! NO RESULTS!!!')
            return
        else:
            # lets use lowest energy for each cluster
            low_index = np.array([index[0] for index in index_sorted2])
            nodes = np.array(self.uuid)
            energy = np.array(self.energies)        
            nodes = nodes[low_index]
            energy = energy[low_index]            
            uuid_ = []
            energy_diff = [abs(en-energy[0]) for en in energy]
            for uuid, edif in zip(nodes, energy_diff):
                if edif <= energy_threshold:
                    uuid_.append(uuid)
        return uuid_
    
    def distint_energy(self, energy_threshold=1.1):
        """Returns symmetry distinct sites energy
        """
        index_sorted2 = self.cluster_muon()
        if len(index_sorted2)==0:
            print('NO CALCULATIONS! NO CLUSTER!! NO RESULTS!!!')
            return
        else:
            # lets use lowest energy for each cluster
            low_index = np.array([index[0] for index in index_sorted2])
            nodes = np.array(self.uuid)
            energy = np.array(self.energies)        
            nodes = nodes[low_index]
            energy = energy[low_index]            
            energy_ = []
            energy_diff = [abs(en-energy[0]) for en in energy]
            for uuid, edif, ener in zip(nodes, energy_diff, energy):
                if edif <= energy_threshold:
                    energy_.append(ener)
        return energy_      

