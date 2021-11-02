#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import numpy as np
from tabulate import tabulate
from scipy.optimize import linear_sum_assignment


# In[2]:


def group_values(calc_values):
    """Here, we pass the nd.array of total fields with
    their labels, index and total field value
    e.g cost_array =  np.array([['$B_{\\mu}^{s}$', '1-1', '0.9694245004388192'],
                                ['$B_{\\mu}^{s}$', '2-1', '0.3696106159894605'],
                                ['$B_{\\mu}^{s}$', '2-2', '0.5802625057445634'],
                                ['$B_{\\mu}^{s}$', '2-3', '1.0119281898999035'],
                                ['$B_{\\mu}^{s}$', '3-1', '2.0119281898999035']]
                                )
    Returns:
          cost array matrix and it is size
    """
    import copy
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

    
    calc_values = np.array(calc_values)
    if len(calc_values)==1:
        return {1: np.array([np.float_(calc_values[:,2])])}
    
    index = calc_values[:,1]
    index_pair = [(inx.split('-')[0], inx.split('-')[1]) for inx in index]
    index_single1 = [] 
    index_single2 = [] 
    for inx in index_pair: 
        index_single1.append(inx[0]) 
        index_single2.append(inx[1])
    
    index_single1_ = list(np.int_(index_single1))  
    calc_values_ = np.float_(calc_values[:,2])
    #print(calc_values_)
    group_indexes = group(index_single1_) 
    group_indexes_index = list(group_indexes.values())
    
    calc_values_list = [[] for i in range(len(group_indexes))]
    calc_values_dict = {}
    dic = {}
    for items in group_indexes.items():
        dic[items[0]] = calc_values_[np.array(items[1])]
    return dic


# In[3]:


class MergeField(object):
    """Function to merge a list of values based on a threshold
    """
    
    @staticmethod
    def index_of_items_from_lists(list_lists, list_items):
        """
        """
        return [list(list_lists).index(item) for item in list_items]

    def __init__(
        self,
        total_fields,
        muon_sites,
        uuid_index=1,
        threshold=0.1
    ):
        """
        Parameters
        ----------
        total_fields : list
            list of calculated field
        muon_sites : nd.array
            muon sites positions
        uuid_index : int
            calculation index, Default = 1
        threshold : float
            
        """
        self.total_fields = np.array(total_fields)
        self.muon_sites = muon_sites
        self.uuid_index = uuid_index
        self.threshold = threshold
        
        if len(self.total_fields) == 1:
            self.index  = self.total_fields[:,1]
            self.labels = list(set(self.total_fields[:,0]))
            self.fields = [np.float_(self.total_fields[:,2])]
        else:
            self.index  = self.total_fields[:,1]
            self.labels = list(set(self.total_fields[:,0]))
            self.fields = list(np.float_(self.total_fields[:,2]))
        
        #self.init_musite_index 
        for i, ind in enumerate(self.index):
            split_ = ind.split('-')
            if split_[-1] == '1':
                self.init_musite_index = i
                #self.init_site_index.append(i)
        self.init_musite_value = self.fields[self.init_musite_index]
#         print('init_musite_index = {} and  init_musite_value = {}'.
#               format(self.init_musite_index, 
#                      self.init_musite_value
#                     )
#              )
        self.length_of_data = len(self.fields)
        original_indexes = [i+1 for i in range(self.length_of_data)]
        
        self.merge_fields_average = {}
        self.merge_indexes_sublists = {}
        
        self.fields_distinct_dic = {}
        self.group_index_sublist_dict = {}
            
        
    def merge_fields(self):
        """
        """
        if self.length_of_data == 1:
            self.merge_fields_average = {'1*':self.fields[0]}
            self.merge_indexes_sublists = {1:[0]}  
            return self.merge_fields_average, self.merge_indexes_sublists
        
        fields_data = np.array(self.fields)
        idx0 = np.argsort(fields_data)
        fields = fields_data[idx0]
        idx = np.argsort(fields)
        diff = np.diff(fields)
        #print('idx0 = ', idx0, 'idx = ', idx)
        avg = fields[:-1]+np.diff(fields)/2
        merge = diff/avg < self.threshold
        # create a list of lists, put the first value of the source data in the first
        lists = [[fields[0]]]
        for i, x in enumerate(fields[1:]):
            # if the gap from the current item to the previous is more than delta
            # Note: the previous item is the last item in the last list
            # Note: the '> 1' is the part you'd modify to make it stricter or more relaxed
            if (x - lists[-1][-1]) / avg[i] > self.threshold:
                # then start a new list
                lists.append([])
            # add the current item to the last list in the list
            lists[-1].append(x)

        #print('lists = ', lists)

        sublists = []
        for items in lists:
            sublists.append(self.index_of_items_from_lists(fields_data, items))
        for i, val in enumerate(sublists):
            self.merge_indexes_sublists[i+1] = val    
        
        for i, l in enumerate(lists):
            #print('values = ', l)
            string = str(i+1)
            if self.init_musite_value in l:
                string = string+'*'
            self.merge_fields_average[string] = np.average(l)
        
        #print('self.merge_fields_average = ', self.merge_fields_average)

        return self.merge_fields_average, self.merge_indexes_sublists
    
    def summary(self):
        """
        """
        muon_sites = np.array(self.muon_sites) 
        fields_dict = self.merge_fields()[0]
        group_indexes = self.merge_fields()[1]
        sites = []
        m_sites = []
        fields = []
        number_dict = {}
        for items1, items2 in zip(group_indexes.items(), fields_dict.items()):
            mu_index = items1[1][0]
            if '*' in items2[0]:
                mu_index = self.init_musite_index          
            number_dict[items1[0]] = len(items1[1])
            sites.append(items2[0])
            fields.append(items2[1])
            m_sites.append(muon_sites[mu_index])
        m_sites = np.around(np.array(m_sites), 6)
        tab = zip(sites, m_sites, fields)
        headers = ['##', 'POSITION (x,y,z)', 'NET FIELD (Tesla)']       
            
        print('\n... Total number of {} equiv fields are merged to give {} distinct field ...'.
              format(self.length_of_data, 
                     len(number_dict)
                    )
             )
        print('\n... Each distinct field has nth terms as : {} ...\n'.format(number_dict))
        print(tabulate(tab, headers=headers, tablefmt="github"))
        print('\n\t[*] Means calculated muon site ...\n')
        
        
    def data_object(self):
        """
        """
        fields_dict = self.merge_fields()[0]
        data = [] 
        for i, items in enumerate(fields_dict.items()):
            data.append([str(self.labels[0]),  str(self.uuid_index)+'-'+str(i+1),    items[1]])
        return data


# In[4]:


class HungarianError(Exception):
    pass
 
# Import numpy. Error if fails
try:
    import scipy
    import numpy as np
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise HungarianError("NumPy is not installed.")
    
    
class HungarianAlgorithm(object):
    """
    """
    def __init__(
        self, 
        calculated_values=None, 
        experimental_values=None
    ):
        """Function to perform Hungarian algorithm
        
        Parameters
        ----------
        calculated_values : dict, list
            calculated value
        experimental_values : list
            experimental value        
        Returns
        -------
        """
        
        if calculated_values is not None and experimental_values is not None:
            exp_values = list(experimental_values)
            argsort = np.argsort(exp_values)
            self._exp_values = list(np.array(exp_values)[argsort]) 
            if isinstance(calculated_values, dict):
                #calc_keys = list(calculated_values.keys())
                self._dict = calculated_values.copy()
                data_type = 'dict'
                calc_values = list(calculated_values.values())
                calc_values = [item for sublist in calc_values for item in sublist]
            elif isinstance(calculated_values, list):
                data_type = 'list'
                calc_values = list(calculated_values)
                #calc_keys = [i+1 for i in range(len(calc_values))]
            elif isinstance(calculated_values, np.ndarray):
                data_type = 'list'
                calc_values = [item for sublist in calc_values for item in sublist]
                #calc_keys = [i+1 for i in range(len(calc_values))]            
            # sort values
            argsort = np.argsort(calc_values)
            self._calc_values = list(np.array(calc_values)[argsort])
            self._data_type = data_type
            #self._calc_keys = calc_keys
            # Results from algorithm.
            self._results = []               # for hungarian results
            self._results2 = []               # return the exact results
            self._PotentialValues = []       # for hungarian results
            self._PotentialValues2 = []       # return the exact results
        else:
            self._calc_values = None
            self._exp_values = [0.0]

    def get_results(self):
        """Get results after calculation."""
        return self._results
 
    def get_potential_values(self):
        """Returns expected value after calculation."""
        #print('Hungarian algorithm gives...')
        if len(self._exp_values) > len(self._PotentialValues):
            print('ALL EXPERIMENTAL DATA CAN\'T BE ASSIGNED...')
        else:
            print('ALL EXPERIMENTAL DATA CAN BE ASSIGNED...')
        print('BY HUNGARIAN ALGORITHM TO GIVE...')
        for i, value in enumerate(self._PotentialValues):
            if self._data_type == 'dict':
                print('VALUE  #{} :: = {} Tesla from calculation {}.'.
                      format(i+1, 
                             value, 
                             self.check_index(value=value)
                            )
                     )
                self.check_values(value=value)        
            else:
                print('VALUE  #{} :: = {} Tesla.'.format(i+1, value))          
        return self._PotentialValues
    
    def __cost_matrix(
        self, 
        calculated_values,
        experimental_values        
    ):
        nrows = len(experimental_values)
        ncols = len(calculated_values)
        self._cost_matrix = np.zeros([nrows, ncols]) 
        self._real_matrix = np.zeros([nrows, ncols])
        for i, val in enumerate(experimental_values):
            self._cost_matrix[i] = [np.abs(val-c) for c in calculated_values] 
            self._real_matrix[i] = [np.abs(0-c) for c in calculated_values]
        return self._cost_matrix, self._real_matrix
      
    
    def check_index(self, value):
        """Found the index of value from a group
        """
        for key, val in self._dict.items():
            for v in list(val):
                if v==value:
                    return key
            
    def check_values(self, value):
        """Found a value from a group
        """
        for key, val in self._dict.items():
            for v in list(val):
                if v==value:
                    if len(val) > 1:
                        print('\tCONTAINS {} SYMMETRY REPLICAS...'.format(len(val)))
                        print('\tWITH VALUES = {} Tesla.'.format(list(val)))
                    
        
    def calculate(
        self, 
        calculated_values=None,
        experimental_values=None
    ):
        """Perform the Hungarian algorithm.
        
        Parameters
        ----------
        calculated_values : dict, list
            calculated value
        experimental_values : list
            experimental value        
        Returns
        -------
        """
        
        if calculated_values is None and self._calc_values is None:
            raise HungarianError("values is invalid or not given")
        elif calculated_values is not None:
            self.__init__(
                calculated_values=calculated_values, 
                experimental_values=experimental_values
            ) 
        calc_values = self._calc_values
        exp_values = self._exp_values
        data_type = self._data_type
        cost_matrix_, real_matrix_ = self.__cost_matrix(calc_values, exp_values)
        
        # Hungarian scipy optimization
        matched_rows, matched_columns = linear_sum_assignment(cost_matrix_)
        
        # Save Results
        for result in zip(matched_rows, matched_columns):
            row, column = result
            rc = (int(row), int(column))
            self._results.append(rc)
            self._PotentialValues.append(real_matrix_[row, column])


# In[ ]:




