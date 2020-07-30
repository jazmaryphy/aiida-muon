import numpy as np

######
from aiida.orm import load_node


def get_fields(uuid, mypatstructure):
    from aiida.orm import Group, load_node
    from muesr.core import Sample
    from muesr.engines import locfield, find_largest_sphere
    
    hf_calc  = load_node(uuid)
    
    o_struct = hf_calc.inputs.parent_folder.creator.inputs.structure
    print("Original structure: ", o_struct.pk) 
    print("   Used in ", *list(o_struct.get_outgoing())
        
    
    ase_structure = hf_calc.outputs.pw__output_trajectory.creator.inputs.structure.get_ase()
    
    # k is 0 by definition
    k = [0,0,0]
    moments = hf_calc.outputs.pw__output_trajectory.get_array('atomic_magnetic_moments')
    
    # Find collinear axis
    aaa =mypatstructure.site_properties['magmom'][0]
    cif_moments, direction = aaa.get_consistent_set_and_saxis(mypatstructure.site_properties['magmom'])
    
    n_atoms = len(ase_structure)
    # go back to 3D structure and make FC complex
    moments = ((np.tile(direction, [n_atoms,1] ).T*moments).T)*(1.+0.j)
    # last is muon, set it to 0
    moments[-1] *= 0.
    
    ContactTesla = hf_calc.outputs.contact_field.value
    
    # Now muesr
    s = Sample()
    s.cell = ase_structure
    s.new_mm()
    # s.mm.k is already 0
    s.mm.fc = moments
    
    s.add_muon(ase_structure.get_scaled_positions()[-1]) 
    
    sc = [10,10,10]
    r = locfield(s,'s', sc, find_largest_sphere(s,sc))[0]
    
    print("Field is")
    print('D', r.D)
    print('L', r.L)
    print('C', direction*ContactTesla)
    print('Tot',r.D + r.L + direction*ContactTesla)

