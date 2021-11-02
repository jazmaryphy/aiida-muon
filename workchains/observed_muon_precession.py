#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


muon_gyro = 135.53882 # MHz/Tesla


# In[3]:


def sort_dictionary_key_and_value_list(test_dict):
    """
    """
    res = dict()
    for key in sorted(test_dict):
        res[key] = sorted(test_dict[key])
    return res


# Data for 16 materials:
# <br>
# Observed muon processsion  frequency in MHz : 'experimental_field_MHz' and 
# <br>
# Converted to Tesla in : 'experimental_field_Tesla'
# <br>
# Continue adding for more data ...

# In[4]:


experimental_field_MHz        =    {"Fe":        [35.92],    # added for checking
                                    "Fe2O3":     [209, 225],
                                    "CoF2":      [31],
                                    "BaFe2As2":  [7, 28.4],
                                    "LaFeAsO":   [3, 28],
                                    "LiCoPO4":   [41.20, 48],
                                    "Cr2S3":     [32.20, 39.0],
                                    "LaMnO3":    [84.90, 128.8],
                                    "MnF2":      [151.8, 1287.25],
                                    "LiMnPO4":   [70, 80], 
                                    "Li2MnO3":   [23.8, 43.8],
                                    "NiO":       [61.3],
                                    "MnO":       [155],
                                    "CoO":       [53.9, 78.4, 150],
                                    "La2NiO4":   [2, 36],
                                    "V2O3":      [15],
                                    "CoAl2O4":   [14.5],                                   
                                   }


experimental_field_MHz  = sort_dictionary_key_and_value_list(experimental_field_MHz)
experimental_field_Tesla = experimental_field_MHz.copy()
experimental_field_Tesla = {k: [v/muon_gyro for v in val] for k, val in experimental_field_Tesla.items()}


# In[5]:


#experimental_field_MHz


# In[6]:


#experimental_field_Tesla


# In[ ]:




