#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import numpy as np
from aiida.engine import submit, run, launch
from pymatgen.core.structure import Structure
from aiida.orm import load_node, Code, Str, Float, Bool, Group, List, Int
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory


# In[2]:


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


def restart_calc(
    uuid,  
    group, 
    input_namelists={}
):
    """
    Restart a failed calculation expecially those exceeds wall time (Error:400) 
    and convergence (Error:410)
    
    uuid : int
        A AiiDA identifier of the calculation to restart
    group : str
        The name of the group to store AiiDA nodes on AiiDA database. Its advisable to keep track
        of this name to make sure you organize your data
    input_namelists : dict
        A user define Quantum ESPRESSO (QE) input namelists. Default={}  
        
    Returns
    -------
    Submit calculation to AiiDA daemon 
    """

    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options 
    
    failed_calculation = load_node(uuid)
    exit_status = failed_calculation.exit_status
    restart_builder = failed_calculation.get_builder_restart()
    parameters_dict = restart_builder.parameters.get_dict()
    parameters_dict['CONTROL']['restart_mode'] = 'restart'
    
    if exit_status == 400:
        """ERROR_OUT_OF_WALLTIME
        The calculation stopped prematurely because it ran out of walltime.
        """         
        parameters_dict = parameters_dict
    if exit_status == 410:
        """ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED
        The electronic minimization cycle did not reach self-consistency.
        """
        parameters_dict['ELECTRONS']['electron_maxstep'] *= 2
        #parameters_dict['ELECTRONS']['conv_thr'] = 1.0e-6
        #parameters_dict['ELECTRONS']['mixing_beta'] *= 1.25
        #parameters_dict['ELECTRONS']['mixing_mode'] = 'plain'
        

    Dict = DataFactory('dict')
    
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
        
        parameters_dict = merge(input_namelistss, parameters_dict) 
       
    restart_builder.parameters = Dict(dict=parameters_dict)
    restart_builder.parent_folder = failed_calculation.outputs.remote_folder
    restart_builder.metadata.label = 'Restart from PwCalculation<{}>'.format(failed_calculation.pk)

    
    clean_workdir = False
    final_scf = False
    if clean_workdir:
        builder.clean_workdir = Bool(True)

    if final_scf:
        builder.final_scf = Bool(True)

    calc = submit(restart_builder)
    if not (group is None):
        group.add_nodes(calc)
    return calc


# In[ ]:




