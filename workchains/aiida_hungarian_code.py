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
    their labels, index and total field value to group the values
    e.g calc_values = np.array([['$B_{\\mu}^{s}$', '1-1', '0.9694245004388192'],
                                ['$B_{\\mu}^{s}$', '2-1', '0.3696106159894605'],
                                ['$B_{\\mu}^{s}$', '2-2', '0.5802625057445634'],
                                ['$B_{\\mu}^{s}$', '2-3', '1.0119281898999035'],
                                ['$B_{\\mu}^{s}$', '3-1', '2.0119281898999035']]
                                )
        and returns a dictionary values based on their index
        {1: np.array([0.9694245]),
         2: np.array([0.36961062, 0.58026251, 1.01192819]),
         3: np.array([2.01192819])}

                                
    Parameters
    ----------
    calc_values : numpy.ndarray
        array containing the label, index and value of field contribution
    Returns: dict
        A dictionary of group values
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


class MergeField:
    """This class is use to Merge/Collapse a set list of values that their
    difference is less than a threshold into unique set of values .
    """
    
    @staticmethod
    def index_of_items_from_lists(list_lists, list_items):
        """Find index of items in a lists
        
        Parameters
        ----------
        list_lists : list
            list of values
        list_items : list
            list of values to identify index from list_lists
        Returns
        -------
        list
        """
        return [list(list_lists).index(item) for item in list_items]

    @staticmethod
    def store_dict_to_file(data_object, filename, uuid_index):
        """To store a data object
        
        Parameters:
        -----------
        data_object : numpuy.ndarray
            A data objec to store in txt file (just for fun to make it editable)
        filenmae : str
            name of file to store data
        uuid_index : int
            index for each file
        """
        import pprint
        dic = data_object.copy()
        with open(filename, "w") as f:
            f.write(pprint.pformat(dic, indent=4))
   

    def __init__(
        self,
        total_fields,
        muon_sites,
        uuid_index = 1,
        threshold = 0.1,
        filename = None
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
        filename : str
            file name to save data object
            
        """
        self.total_fields = np.array(total_fields)
        self.muon_sites = muon_sites
        self.uuid_index = uuid_index
        self.threshold = threshold
        self.filename = filename   
        
        if filename is None or filename == "":
            self.filename = "data"
        else:
            self.filename = filename
        
        if len(self.total_fields) == 1:
            self.index  = self.total_fields[:,1]
            self.labels = list(set(self.total_fields[:,0]))
            self.fields = [np.float_(self.total_fields[:,2])]
        else:
            self.index  = self.total_fields[:,1]
            self.labels = list(set(self.total_fields[:,0]))
            self.fields = list(np.float_(self.total_fields[:,2]))
        
        # The self.init_musite_index and self.init_musite_value
        # to keep note of the index of calculated field (not replicas)
        # and its value 
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
        """ Perform the merging of values
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
            # if the gap from the current item to the previous is more than the threshold
            # Note: the previous item is the last item in the last list
            # Note: the '> self.threshold' is the part you'd modify to make it stricter 
            # or more relaxed
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

        return self.merge_fields_average, self.merge_indexes_sublists
    
    def summary(self):
        """ Display the summary of informations
        """
        muon_sites = np.array(self.muon_sites)
        fields_dict = self.merge_fields()[0]
        group_indexes = self.merge_fields()[1]

        filename = self.filename+'_merge_index_calc_'+str(self.uuid_index)+'.txt'
        print('... Saving complete merge index\'s for calculation #{} to a file : {}'.
              format(self.uuid_index,
                     filename
                    )
             )
        self.store_dict_to_file(group_indexes, filename, self.uuid_index)

        sites = []
        m_sites = []
        fields = []
        number_dict = {}
        multiplicity = []
        for items1, items2 in zip(group_indexes.items(), fields_dict.items()):
            mu_index = items1[1][0]
            if '*' in items2[0]:
                mu_index = self.init_musite_index
            number_dict[items1[0]] = len(items1[1])
            sites.append(items2[0])
            fields.append(items2[1])
            m_sites.append(muon_sites[mu_index])
            multiplicity.append(len(items1[1]))
        m_sites = np.around(np.array(m_sites), 6)
        tab = zip(sites, m_sites, fields, multiplicity)
        headers = ['##', 'POSITION (x,y,z)', 'NET FIELD (Tesla)', 'MULTIPLICITY ##']
            
        print('\n... Total number of {} equiv fields are merged to give {} distinct field ...\n'.
              format(self.length_of_data, 
                     len(number_dict)
                    )
             )
#         print('\n... Each distinct field has nth terms as : {} ...\n'.format(number_dict))
        print(tabulate(tab, headers=headers, tablefmt="github"))
        print('\n\t[*] Means calculated muon site ...\n')
        
        
    def data_object(self):
        """Prepare the merge values into a form of input data of 'total_fields'
        """
        fields_dict = self.merge_fields()[0]
        data = [] 
        for i, items in enumerate(fields_dict.items()):
            data.append([str(self.labels[0]),  str(self.uuid_index)+'-'+str(items[0]),    items[1]])
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
    raise HungarianError("numpy or scipy not installed.")
    
    
class HungarianAlgorithm:
    """This class perfrom a Hungarian algorithm to assign experimental values
        to calculated values
    """
    
    
    @staticmethod
    def __cost_matrix(
        calculated_values,
        experimental_values        
    ):

        """Calculate the cost and real matrix. The cost matrix is weight of task to
        assign each experimental values to any calculated values, which is absolute
        difference between experimental and calculated values. The real matrix 
        is weight of calculated value only.
        
        Parameters
        ----------
        calculated_values : dict, list
            calculated value
        experimental_values : list
            experimental value     
            
        Returns
        -------
        tuple
        cost_matrix : numpy.ndarray
            the cost matrix to optimized
        real matrix :  numpy.ndarray
            the real matrix of the calculated values
        """
        nrows = len(experimental_values)
        ncols = len(calculated_values)
        cost_matrix = np.zeros([nrows, ncols]) 
        real_matrix = np.zeros([nrows, ncols])
        for i, val in enumerate(experimental_values):
            cost_matrix[i] = [np.abs(val-c) for c in calculated_values] 
            real_matrix[i] = [np.abs(0-c) for c in calculated_values]
        return cost_matrix, real_matrix        
    
    def __init__(
        self, 
        calculated_values=None, 
        experimental_values=None
    ):
        """This class perfrom a Hungarian algorithm to assign experimental values
        to calculated values
        
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
        match_data_object = {}
        for i, value in enumerate(self._PotentialValues):
            if self._data_type == 'dict':
                other_values, index = self.check_other_values_and_index(value=value)
                print('VALUE  #{} :: = {} in Tesla from calculation {}.'.
                      format(i+1, 
                             value, 
                             index
                            )
                     )
                print('\tCONTAINS {} SYMMETRY REPLICAS...'.format(len(other_values)))
                print('\tWITH VALUES = {} in Tesla.'.format(other_values))      
            else:
                print('VALUE  #{} :: = {} Tesla.'.format(i+1, value))
            match_data_object[i+1] = value
        print('\nMATCH DATA OBJECT = {}\n'.format(match_data_object))
        return self._PotentialValues
         
    def check_other_values_and_index(self, value):
        """Found a value from a group
        """
        for key, val in self._dict.items():
            if value in val:
                return val, key                    
        
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




