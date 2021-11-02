#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hashlib
import warnings
import numpy as np
from aiida import load_profile
import argparse, os, yaml, json
from pymatgen.core import Structure
from aiida.orm import load_node, load_group
from aiida_muon_cluster import GenerateCluster
from aiida_muon_restart_calc import restart_calc
from aiida_muesr_getfield import EstimateFieldContributions
from aiida_hungarian_code import group_values, HungarianAlgorithm, MergeField
from observed_muon_precession import experimental_field_Tesla as observed_field_dict
from aiida_muon_query import QueryCalculationsFromGroup, QueryNodeEnergyPositions, ratio_of_magmom
from aiida_muon_utils import generate_supercells, run_wc, check_group_elements, run_hyperfine, run_calculation

###from aiida_muesr_utils import get_fields
###from aiida_muesr_getfield_old import get_fields
###from print_table import print_table

try:
    input = raw_input
except NameError:
    pass


# In[2]:


def restart_supercells(
    uuid_list,
    group_name,
    input_namelists
):
    """Restart PwCalculculations having the same error messagess
    """
    from aiida.orm import Group # aiida stuff should not be here I think
    g, just_created = Group.objects.get_or_create(group_name)

    for l, uuid in enumerate(uuid_list):
        restart_calc(uuid,
                     g,
                     input_namelists=input_namelists
                    )

def gen_supercells(
    structure,
    grid_size,
    sc_size,
    imp_method,
    positions=None
):
    """
    """
    return generate_supercells(
        structure,
        grid_size, 
        supercell=sc_size, 
        method=imp_method, 
        positions=positions
    ) 


def run_supercells(
    supercells, 
    NAME, 
    group_name, 
    codename, 
    pseudo_family, 
    mu_plus, 
    code_runtime_options, 
    input_namelists
):
    """Submits the workchains for all supercells.
    """
    from aiida.orm import Group # aiida stuff should not be here I think
    g, just_created = Group.objects.get_or_create(group_name)
    
    for l, supercell in enumerate(supercells):
        run_wc(
            l, 
            NAME, 
            supercell, 
            g, 
            codename = codename, 
            pseudo_family = pseudo_family,
            input_namelists = input_namelists, 
            mu_plus = mu_plus, 
            code_runtime_options = code_runtime_options
        )
        

def run_hyperfines(
    nodes,
    structure,
    sc_size,
    group_name,
    pp_codename,
    projwfc_codename,
    symprec,
    if_equivalent_sites,
    code_runtime_options
):
    """
    """
    for i, pk in enumerate(nodes):
        run_hyperfine(
            uuid = pk, 
            structure = structure, 
            sc_size = sc_size, 
            group = group_name, 
            pp_codename = pp_codename,
            projwfc_codename = projwfc_codename,
            symprec = symprec,
            if_equivalent_sites = if_equivalent_sites,
            if_with_distortions = if_with_distortions,
            code_runtime_options = code_runtime_options
        )
               
def run_fields(
    nodes, 
    structure, 
    sc_size, 
    if_pristine,
    if_scale_moment,
    if_with_contact,
    if_scale_contact,
    if_equivalent_sites,
    if_with_distortions,
    field_merge_threshold,
    file_name
):
    """
    """       
    tot_obj = []
    tot_merge_obj = []    
    for i, pk in enumerate(nodes):
        EFC = EstimateFieldContributions(uuid = pk,
                                         structure = structure,
                                         sc_size = sc_size,
                                         uuid_index = i+1,
                                         if_pristine = if_pristine,
                                         if_scale_moment = if_scale_moment,
                                         if_with_contact = if_with_contact,        
                                         if_scale_contact = if_scale_contact,
                                         if_equivalent_sites = if_equivalent_sites,
                                         if_with_distortions = if_with_distortions,
                                         file_name = file_name
                                        )
        print('')
        EFC.print_logger
        EFC.save_muon_sites()

        EFC.calculate()

        M = MergeField(total_fields=EFC.get_total_object(),
                       muon_sites=EFC.get_muon_sites,
                       uuid_index=i+1,
                       threshold=field_merge_threshold
                      )

        M.summary()
        tot_obj.append(M.data_object())

    tot_obj = np.array([item for sublist in tot_obj for item in sublist])

    # I need experimental_values for each materials
    experimental_values = observed_field_dict.get(file_name, None)
    print('\n Performing Hungarian matching of ({}) experimental data \n'.format(experimental_values))
    tot_obj = group_values(tot_obj)
    hungarian = HungarianAlgorithm(tot_obj, experimental_values)
    hungarian.calculate()
    hungarian.get_potential_values()
        
def load_bulk_structure(file_path):
    return Structure.from_file(file_path)

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

def save_to_file(filename, output):
    with open(filename, "w") as f: 
        json.dump(output,f) 
            
def read_qe_inputs(args):
    """to read Inputs of QE
    """
    import ast
    input_file=args
    with open(input_file) as f:
         inputs = f.read()
    return ast.literal_eval(inputs)
    
def boolean_string(s):
    """If we  must parse a string with True or False.
    https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
    
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


# In[3]:


if __name__ == "__main__":
    import argparse
    load_profile()

    parser = argparse.ArgumentParser(description='Run muon simulation')
    parser.add_argument('CASE', type=str, help='CASE file')

    parser.add_argument('--structure', type=str, default=argparse.SUPPRESS, 
                        help='structure file')
    parser.add_argument('--title', type=str, default=argparse.SUPPRESS, 
                        help='title of the simulation')
    parser.add_argument('--upf-family', type=str, default=argparse.SUPPRESS, 
                        help='pseudo')
    parser.add_argument('--group', type=str, default=argparse.SUPPRESS, 
                        help='group')
    parser.add_argument('--sc-size', type=str, default=argparse.SUPPRESS, 
                        help='tess, voronoi')
    parser.add_argument('--sc-gen-mode', type=str, default=argparse.SUPPRESS, 
                        help='tess, voronoi, random, ugrid')
    parser.add_argument('--pw-code', type=str, default=argparse.SUPPRESS, 
                        help='pw')
    parser.add_argument('--pp-code', type=str, default=argparse.SUPPRESS, 
                        help='pp')
    parser.add_argument('--projwfc-code', type=str, default=argparse.SUPPRESS, 
                        help='projwfc')
    parser.add_argument('--runtime-options', type=str, default=argparse.SUPPRESS, 
                        help='pw options')
    parser.add_argument('--charged', type=boolean_string, default=argparse.SUPPRESS, 
                        help='True|False')
    parser.add_argument('--if_equivalent_sites', type=boolean_string, default=False, 
                        help='True|False')
    parser.add_argument('--if_with_contact', type=boolean_string, default=False, 
                        help='True|False')    
    parser.add_argument('--if_scale_moment', type=boolean_string, default=True, 
                        help='True|False')
    parser.add_argument('--if_scale_contact', type=boolean_string, default=False, 
                        help='True|False')
    parser.add_argument('--if_with_distortions', type=boolean_string, default=True,
                        help='True|False')  
    parser.add_argument('--file_name', type=str, default=argparse.SUPPRESS, 
                        help='name of file to read experimental data')
#     parser.add_argument('--charged', type=bool, default=argparse.SUPPRESS, 
#                         help='True|False')
    parser.add_argument('--inputfile', type=str, default=argparse.SUPPRESS, 
                        help='files to read input namelists QE')
    parser.add_argument('--if_with_energy', type=boolean_string, default=True, 
                        help='True|False')
    parser.add_argument('--if_pristine', type=boolean_string, default=False, 
                        help='True|False')
    parser.add_argument('--energy_threshold', type=float, default=0.01, 
                        help='total energy difference threshold in eV for muon site clustering')
    parser.add_argument('--field_merge_threshold', type=float, default=0.099, 
                        help='threshold in Tesla to merge two or more fields to one')
    parser.add_argument('--symprec', type=float, default=0.01, 
                        help='distance tolerance for symmetry finging in cartesian coordinates')
    parser.add_argument('--frac_coords', type=boolean_string, default=argparse.SUPPRESS, 
                        help='True|False')
    parser.add_argument('--output_file', type=str, default=argparse.SUPPRESS, 
                        help='output file to save fields')    
#     parser.add_argument('--search_equiv', type=boolean_string, default=False,             
#                         help='search equivalent sites, True|False')
#     parser.add_argument('--reduced_sites', type=boolean_string, default=argparse.SUPPRESS, 
#                         help='True|False')
#     parser.add_argument('--reduced_distance', type=float, default=0.0, 
#                         help='distance in angtrom to select equiv muon sites')
    parser.add_argument('--workchain_name', type=str, default=argparse.SUPPRESS, 
                        help='WorkChain name to query data')
    parser.add_argument('--process_status', type=str, default='finished', 
                        help='type of state to query data, "finished" or "excepted" ')
    parser.add_argument('--exit_status', type=str, default='0', 
                        help='exit status number to look for" ')
    
    parser.add_argument('--task', type=str, default=argparse.SUPPRESS, 
                        help='what to do')
    
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
    get_option(case, 'if_equivalent_sites', args)
    get_option(case, 'if_with_contact', args)
    get_option(case, 'if_scale_moment', args)
    get_option(case, 'if_scale_contact', args)
    get_option(case, 'if_with_distortions', args)
    get_option(case, 'file_name', args)
    get_option(case, 'if_with_energy', args)
    get_option(case, 'if_pristine', args)
    get_option(case, 'energy_threshold', args)
    get_option(case, 'field_merge_threshold', args)
    get_option(case, 'symprec', args)
    get_option(case, 'frac_coords', args)
    get_option(case, 'output_file', args)
#     get_option(case, 'search_equiv', args)
#     get_option(case, 'reduced_sites', args)
#     get_option(case, 'reduced_distance', args)
    get_option(case, 'workchain_name', args)
    get_option(case, 'process_status', args)
    get_option(case, 'exit_status', args)
    
 
    if args.task == 'sites':
        structure = load_bulk_structure(case['structure'])
        #to read QE input file namelists and paramaters
        input_namelists = {}
        if args.inputfile:
            #input_namelist = read_qe_inputs(args.inputfile)
            input_namelists = read_qe_inputs(args.inputfile)
        pos = None
        grid_size = None
        gen_mode = case['sc_gen_mode']
        if 'random' in gen_mode.lower() or 'grid' in gen_mode.lower():   
            grid_size = int(gen_mode.lower().split(':')[1])  
            pos =  np.random.random([1,3])   #  [[1.565560718,   0.001440236,   2.328253886]]
            if 'grid' in gen_mode.lower():      
                gen_mode = 'grid'
            else:
                gen_mode = 'manual' 
        
        supercells = gen_supercells(structure, 
                                    grid_size,
                                    [int(x) for x in case['sc_size'].split()],
                                    gen_mode, 
                                    pos
                                   )
        
        print("Generated ", len(supercells))
        #input_namelists = [input_namelist]*len(supercells)
        if yes_or_no("Submit them?"):
            run_supercells(supercells, 
                           case['title'], 
                           case['group'], 
                           case['pw_code'], 
                           case['upf_family'],
                           mu_plus=case['charged'], 
                           code_runtime_options=case['runtime_options'],
                           input_namelists=input_namelists
                          )
            
    elif 'restart_sites' in args.task:        
        t, uuid = args.task.split('->')
        input_namelists = {}
        if args.inputfile:
            input_namelists = read_qe_inputs(args.inputfile)        
        calc=load_node(uuid)
        # group has no process type
        if calc.process_type is None or calc.process_type == "": 
            g = load_group(uuid)
            process_status_name = case['process_status']
            Q = QueryCalculationsFromGroup(group_uuid=uuid, 
                                           process_label="PwRelaxWorkChain", 
                                           process_status_name=process_status_name
                                          )
            
            r_uuid = Q.query_pw_nodes_with_exit_status(case['exit_status'])
            Q.query_nodes_tabulate()
            print("\n uuid found : {}".format(len(r_uuid)))
            if yes_or_no(" Run calculations?"):
                restart_supercells(uuid_list = r_uuid,
                                   group_name = g.label,
                                   input_namelists = input_namelists
                                  )
                #
                
        else:
            # Here we are sure parsing a single node of "PwCalculation"
            restart_supercells(uuid_list = [uuid],
                               group_name = case['group'],
                               input_namelists = input_namelists
                              )            
        
    elif args.task == 'list':
        data = {'E':[], 'hash':[], 'pk': [], 'type': [], 'state':[]}
        for element in check_group_elements(case['group']):
            if ('Relax' in element.attributes['process_label']) and             (element.attributes['process_state'] == 'finished'):
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
                    header=[ "Calc Type", "Calc PK", "Energy", "State", "Inputs hash"],   
                    wrap=True, max_col_width=20, wrap_style='wrap', row_line=True)
        
    elif 'unitcell' in args.task:
        if yes_or_no("Submit them?"):
            input_namelists = {}
            #to read QE input file namelists and paramaters
            if args.inputfile:
                input_namelists = read_qe_inputs(args.inputfile)  
            structure = load_bulk_structure(case['structure'])
            run_calculation(name=case['title'], 
                            struct=structure, 
                            group=case['group'], 
                            codename=case['pw_code'], 
                            pseudo_family=case['upf_family'], 
                            input_namelists=input_namelists, 
                            code_runtime_options=case['runtime_options']
                           ) 
            
    elif 'hyperfine' in args.task:
        #Here we run hyperfine calculation(s) for relaxed structure(s)
        t, uuid=args.task.split('->')
        calc=load_node(uuid)
        structure = load_bulk_structure(case['structure'])
        # group has no process type
        if calc.process_type is None or calc.process_type =="": 
            # For all relaxed results in a group
            Q = QueryCalculationsFromGroup(group_uuid=uuid, 
                                           process_label="PwRelaxWorkChain"
                                          )
            
            sr_uuid = Q.get_all_relaxed_with_exit_status(case['exit_status'])
            print("\n uuid found : {}".format(len(sr_uuid)))
            if yes_or_no(" run hyperfine?"):
                """run hyperfine calculation for all the nodes in the group"""    
                run_hyperfines(nodes = sr_uuid, 
                               structure = structure, 
                               sc_size = case['sc_size'], 
                               group_name = case['group'], 
                               pp_codename = case['pp_code'], 
                               projwfc_codename = case['projwfc_code'],
                               symprec =float(case['symprec']),
                               if_equivalent_sites = case['if_equivalent_sites'], 
                               if_with_distortions = case['if_with_distortions'],
                               code_runtime_options = None
                              )
        else:
            # Here we are sure parsing a single node of "PwCalculation"
            run_hyperfines(nodes = [uuid], 
                           structure = structure, 
                           sc_size = case['sc_size'], 
                           group_name = case['group'], 
                           pp_codename = case['pp_code'], 
                           projwfc_codename = case['projwfc_code'],
                           symprec = float(case['symprec']),
                           if_equivalent_sites = case['if_equivalent_sites'], 
                           if_with_distortions = case['if_with_distortions'],
                           code_runtime_options = None
                          )            

    elif 'ratio' in args.task:
        t, uuid=args.task.split('->')
        ratio_of_magmom(uuid, load_bulk_structure(case['structure']))  
        
    elif 'fields' in args.task:
        # Here, we calculate local field contributions
        t, uuid = args.task.split('->')
        task = t.split('|')[-1]
        structure = load_bulk_structure(case['structure'])
        calc=load_node(uuid)
        # group has no process type
        if calc.process_type is None or calc.process_type == "":
            # For all results with hyperfine calculations
            if case['if_with_contact']:                
                Q = QueryCalculationsFromGroup(group_uuid=uuid, 
                                               process_label="PwContactFieldWorkChain", 
                                               process_status_name=case['process_status']
                                              )
                nodes = Q.get_hyperfine_nodes()
            else:
                # In case hyperfine calculations is not needed 
                process_label = 'PwRelaxWorkChain'
                G = GenerateCluster(group_uuid=uuid,
                                    structure=structure, 
                                    sc_size=case['sc_size'], 
                                    process_label=process_label, 
                                    energy_threshold=float(case['energy_threshold']), 
                                    symprec = float(case['symprec']),
                                    if_with_energy=case['if_with_energy']
                                   )
                # We use only symmetry distinct muon site calculation
                nodes =  G.distinct_cluster(energy_threshold=1.0)                
                
            print("\n uuid found : {}".format(len(nodes)))
            if yes_or_no(" Calculate fields?"):
                run_fields(nodes = nodes, 
                           structure = structure,
                           sc_size = case['sc_size'], 
                           if_pristine = case['if_pristine'], 
                           if_scale_moment = case['if_scale_moment'], 
                           if_with_contact = case['if_with_contact'],
                           if_scale_contact = case['if_scale_contact'], 
                           if_equivalent_sites = case['if_equivalent_sites'],                            
                           if_with_distortions = case['if_with_distortions'],
                           field_merge_threshold = case['field_merge_threshold'],
                           file_name = case['file_name']                           
                          )        
    
        else:   
            # calculate field a single calculation
            run_fields(nodes = [uuid], 
                       structure = structure,
                       sc_size = case['sc_size'], 
                       if_pristine = case['if_pristine'], 
                       if_scale_moment = case['if_scale_moment'], 
                       if_with_contact = case['if_with_contact'],
                       if_scale_contact = case['if_scale_contact'], 
                       if_equivalent_sites = case['if_equivalent_sites'],                            
                       if_with_distortions = case['if_with_distortions'],
                       field_merge_threshold = case['field_merge_threshold'],
                       file_name = case['file_name']                           
                      ) 
            
    elif 'query' in args.task:
        query, uuid = args.task.split('->')
        group = load_group(uuid)
        t = query.split("|")[-1]
        if t == "cluster":
            print('\n QUERYING DATA FOR  GROUP "{}",  NAMED "{}" :\n'.format(uuid, group.label))
            process_label = 'PwRelaxWorkChain'
            structure = load_bulk_structure(case['structure']) 
            G = GenerateCluster(group_uuid=uuid, 
                                structure=structure, 
                                sc_size=case['sc_size'],  
                                process_label=process_label, 
                                energy_threshold=float(case['energy_threshold']), 
                                symprec=float(case['symprec']),
                                if_with_energy=case['if_with_energy']
                               )
            G.tabulate_cluster()   
            print('')
            print('distinct_nodes =',  G.distinct_cluster())
            print('')
        else:
            if case['workchain_name'] == 'relax':
                process_label = 'PwRelaxWorkChain'
            if case['workchain_name'] == 'contact':
                process_label = 'PwContactFieldWorkChain'
            process_status_name = case['process_status']
            print('\n QUERYING "{}" NODES FOR "{}" FOR  GROUP "{}",  NAMED "{}" :\n'.format(process_status_name.upper(), 
                                                                                            process_label, uuid, group.label))
            Q = QueryCalculationsFromGroup(group_uuid=uuid, 
                                           process_label=process_label, 
                                           process_status_name=process_status_name
                                          )                       
            Q.tabulate_query_nodes()
    else:
        print("Unknown task, Bye.")

    store_case_file(args.CASE, case)

