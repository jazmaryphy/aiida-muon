import argparse, os, yaml
import warnings
from aiida.aiida_muons.workchains.aiida_muon_utils import generate_supercells, run_wc, check_group_elements, run_hyperfine, \
                                                          grid_parameters,  show_structures, show_structures_with_muon
from aiida.aiida_muons.workchains.aiida_muesr_ugrid import generate_uniform_grid, show_cell, show_structure_with_muon
#from aiida.aiida_muons.workchains.aiida_muesr_utils import get_fields
from pymatgen.core import Structure
from aiida import load_profile
from aiida.aiida_muons import print_table
from flask_restplus import inputs

import numpy as np
import hashlib

try:
   input = raw_input
except NameError:
   pass

def gen_supercells(structure, grid_parameters, sc_size, imp_method,  positions=None):

    return generate_supercells(structure, grid_parameters, supercell=sc_size, method=imp_method, positions=positions) 

def run_supercells(supercells, NAME, group_name, codename, pseudo_family, mu_plus, code_runtime_options):
    """
    Submits the workchains for all supercells.
    """
    from aiida.orm import Group # aiida stuff should not be here I think
    g, just_created = Group.objects.get_or_create(group_name)

    for l, supercell in enumerate(supercells):
        run_wc(l, NAME, supercell, g, codename = codename, pseudo_family = pseudo_family,
                mu_plus=mu_plus, code_runtime_options=code_runtime_options)

def load_bulk_structure(file_path):
    return Structure.from_file(file_path)
    
def grid_param(file_path, grid_size):
    return grid_parameters(file_path, grid_size)

def print_aiida_options():
    pass

def load_case_file(filename):
    if not os.path.isfile(filename):
        return {}

    with open(filename) as f:
        # use safe_load instead load
        return yaml.safe_load(f)

def store_case_file(filename, case):
    with open(filename, "w") as f:
        yaml.dump(case, f)

def get_option(case, name, args, can_be_updated=False):
    """
    Returns true if option has been parsed, false if already set by case file.
    """
    value = case.get(name, None)
    if (not value) or can_be_updated:
        if name in args:
            case[name] = args.__getattribute__(name)
            return True
        elif not value:
            warnings.warn('Option {} missing!'.format(name), 
                            RuntimeWarning, stacklevel=0)
            return False
            
    else:
        if name in args:
            warnings.warn('Option {} already stored in CASE file. Input IGNORED!'.format(name), 
                            RuntimeWarning, stacklevel=0)
        return False
    return True

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

if __name__ == "__main__":
    import argparse
    load_profile()

    parser = argparse.ArgumentParser(description='Run muon simulation')
    parser.add_argument('CASE', type=str, help='CASE file')

    parser.add_argument('--structure', type=str, default=argparse.SUPPRESS, help='structure file')
    parser.add_argument('--title', type=str, default=argparse.SUPPRESS, help='title of the simulation')
    parser.add_argument('--upf-family', type=str, default=argparse.SUPPRESS, help='pseudo')
    parser.add_argument('--group', type=str, default=argparse.SUPPRESS, help='group')
    parser.add_argument('--sc-size', type=str, default=argparse.SUPPRESS, help='tess, voronoi')
    parser.add_argument('--sc-gen-mode', type=str, default=argparse.SUPPRESS, help='tess, voronoi, random, ugrid')
    parser.add_argument('--pw-code', type=str, default=argparse.SUPPRESS, help='pw')
    parser.add_argument('--pp-code', type=str, default=argparse.SUPPRESS, help='pp')
    parser.add_argument('--projwfc-code', type=str, default=argparse.SUPPRESS, help='projwfc')
    parser.add_argument('--runtime-options', type=str, default=argparse.SUPPRESS, help='pw options')
    #parser.add_argument('--charged', type=bool, default=argparse.SUPPRESS, help='True/False')  # old code
    parser.add_argument('--charged', type=str, default=argparse.SUPPRESS, help='true|false') # simplest solution I can think of
    
    parser.add_argument('--task', type=str, default=argparse.SUPPRESS, help='what to do')

    

    args = parser.parse_args()
    case = load_case_file(args.CASE)

    get_option(case, 'structure', args)
    get_option(case, 'title', args)
    get_option(case, 'upf_family', args)
    get_option(case, 'group', args)
    get_option(case, 'sc_gen_mode', args, can_be_updated=True)
    get_option(case, 'sc_size', args)
    get_option(case, 'pw_code', args)
    get_option(case, 'pp_code', args)
    get_option(case, 'projwfc_code', args)
    get_option(case, 'runtime_options', args, can_be_updated=True)
    get_option(case, 'charged', args)
    

    if args.task == 'sites':
        structure = load_bulk_structure(case['structure'])
        pos = None
        uniform_grid_parameters = None
        gen_mode = case['sc_gen_mode']
        if  'random' in gen_mode.lower():
            n_rnd = int(gen_mode.lower().split(':')[1])
            pos = np.random.random([1,3])
            gen_mode = 'manual'
            
            
        if 'grid' in gen_mode.lower():   
            mu_position =  []     
            grid_size = int(gen_mode.lower().split(':')[1])
            gen_mode = 'ugrid'
            structure_copy = parser.parse_args()
            structure_copy = structure_copy.structure
            uniform_grid_parameters = grid_param(structure_copy, grid_size) 
            mu_position=generate_uniform_grid(structure_copy, grid_size)
                               
        supercells = gen_supercells(structure, uniform_grid_parameters, \
                                    [int(x) for x in case['sc_size'].split()], \
                                    gen_mode, pos)
        
        if yes_or_no("Show structure?"):
           for mu_i, mu_j in enumerate(mu_position):
               print("Grid position {}, {}".format(mu_i, mu_j))
               show_structures_with_muon(structure_copy,  mu_position=mu_j,  sc_size=(1,1,1))
               #show_structures_with_muon(structure_copy,  mu_position=mu_j,  sc_size=[int(x) for x in case['sc_size'].split()])
             
        print("Generated ", len(supercells) )
        if yes_or_no("Submit them?"):
            run_supercells(supercells, 
                           case['title'], case['group'], case['pw_code'], case['upf_family'],
                           mu_plus=case['charged'], 
                           code_runtime_options=case['runtime_options'])
                           
    elif args.task == 'list':
        data = {'E':[], 'hash':[], 'pk': [], 'type': [], 'state':[]}
        for element in check_group_elements(case['group']):
            if ('Relax' in element.attributes['process_label']) and (element.attributes['process_state'] == 'finished'):
                tra = element.outputs.output_trajectory
                inp = element.inputs
                one_big_md5 = ""
                for key in inp:
                    if key in ['structure', 'settings']:
                        continue
                    one_big_md5 = hashlib.md5((one_big_md5+inp[key].extras['_aiida_hash']).encode('utf-8')).hexdigest()
                
                data['E'].append(tra.get_array('energy')[-1])
                data['hash'].append(one_big_md5)
                data['pk'].append(element.pk)
                data['type'].append(element.attributes['process_label'])
                data['state'].append(element.attributes['process_state'])                
            else:
                data['E'].append(0.0)
                data['hash'].append('-')
                data['pk'].append(element.pk)
                data['type'].append(element.attributes['process_label'])
                data['state'].append(element.attributes['process_state'])

            

        print_table(list(zip(data['type'], data['pk'], data['E'], data['state'], data['hash'])),
                    header=[ "Calc Type", "Calc PK", "Energy", "State", "Inputs hash"],   wrap=True, max_col_width=20, wrap_style='wrap', row_line=True)


    elif 'hyperfine' in args.task:
        t, uuid=args.task.split('->')
        run_hyperfine(uuid, case['group'], case['pp_code'], case['projwfc_code'], code_runtime_options=None)

    elif 'fields' in args.task:
        t, uuid = args.task.split('->')

        structure = load_bulk_structure(case['structure'])
        get_fields(uuid, structure)
        
    else:
        print("Unknown task, Bye.")

    store_case_file(args.CASE, case)
