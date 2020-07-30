# imports from python
#from itertools import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import tess



##########################################################
def compute_voronoi(points):                                                        
    """ Function to return the python container having information of Voronoi cells for            
        for given points in 3D space                                                               

    Parameters                                                                                          
        pts = numpy array of points coordinates                                                    

    Returns                                                                                         
        container = python contrainer having information of Voronoi cells                          
    """

    P = np.array(points)
    
    # box limits along x, y, and z axis
    Lx = 50
    Ly = 50
    Lz = 50
    
    limits=[(-50,-50,-50),(50,50,50)]  # two 3-tuples lower and upper limits
    
    cntr = tess.Container(P, limits=limits, periodic=False)#, periodic=(False, False, False))

    return cntr

##########################################################
def calculate_midpoint(p1, p2):
    """ Calculate the midpoint given coordinates
    
    Parameters                                                                                          
        p1, p2 = numpy array of point coordinates                                                  
                                                                                                   
    Returns                                                                                         
        midpoint = numpy array of midpoint coordinates                                             
    """

    return((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0, (p1[2]+p2[2])/2.0)

##########################################################
def calculate_polygon_centroid(poly_pts):
    """ Function to calculate the centroid of non-self-intersecting polygon

    Parameters
        pts = numpy array of coordinates of vertices of polygon

    Returns
        centroid = numpy array of centroid coordinates
    """

    P = np.array(poly_pts)
    C = np.mean(P, axis=0)

    return C

##########################################################
def neighbor_list(list):
    """ Function to form unique neighboring pairs along the polygon perimeter

    Parameters 
        list = list of indicies of Voronoi vertices forming the perimeter                          

    Returns
        list = list of neighboring pairs as tuples
    """

    i = 0
    while i + 1 < len(list):
        yield (list[i], list[i+1])
        i += 1
    else:
        yield (list[i], list[0])

##########################################################
def get_vertices(site_num, cntr):
    """ Function that returns vertices of the Voronoi associated with given site

    Parameters
        site_num = number for the lattice site of interest
        cntr = python contrainer having information of Voronoi cells

    Returns
        vertices = numpy array of Voronoi vertices coordinates
    """

    list_voronoi_vertices = cntr[site_num].vertices()
    V = list_voronoi_vertices

    # convert the list to numpy array
    V = np.asarray(V)

    return V

##########################################################
def get_edgecenter(site_num, cntr):
    """ Function that returns vertices unique edge centers of the Voronoi associated with specific\
 lattice site

    Parameters
        site_num = number for the lattice site of interest
        cntr = python contrainer having information of Voronoi cells

    Returns
        Edge center = numpy array of Voronoi edge center coordinates
    """

    list_face_vertices_indices = cntr[site_num].face_vertices()

    V_vertices = get_vertices(site_num, cntr)

    all_midpoint = []

    for face in list_face_vertices_indices:
        for(x,y) in neighbor_list(face):
            midpoint = calculate_midpoint(V_vertices[x], V_vertices[y])
            all_midpoint.append(midpoint)

    #using set so to choose only unique edge centers
    S = set(all_midpoint)

    #converting set to list
    Ec = list(S)

    #converting list to numpy array
    Ec = np.asarray(Ec)

    return Ec

##########################################################
def get_facecentroid(site_num, cntr):
    """Function the returns vertices of face centers of the Voronoi associated with specific latti\
ce site

    Parameters
        site_num = number for the lattice site of interest
        cntr = python contrainer having information of Voronoi cells

    Returns
        Face center = numpy array of Voronoi face center coordinates                               
    """

    list_face_vertices_indices = cntr[site_num].face_vertices()

    V_vertices = get_vertices(site_num, cntr)

    list_face_centroid = []

    for face in list_face_vertices_indices:
        l = []
        for j in face:
            vv  = V_vertices[j]
            l.append(vv)
        l = np.asarray(l)
        pc = calculate_polygon_centroid(l)
        list_face_centroid.append(pc.tolist())

    Fc = list_face_centroid

    # converting list to numpy array                                                               
    Fc = np.asarray(Fc)

    return Fc

##########################################################
def get_all_interstitials(bulk_structure):
    """ function to return list of all interstitial sites using Voronoi.py

    Parameters
        prim = pylada primitive structure
        positions = positions (numpy array) to compute interstitials for

    Returns
        Inst_list = list of list, with inner list containing ['atom type', [x,y,z]]
    """

    ints_list = []

    all_nbs = bulk_structure.get_all_neighbors(r=10)
    for site_num in range(bulk_structure.num_sites):
        site = bulk_structure.cart_coords[site_num]
        

        points = [site]

        ### creating list with site and its neighbors
        for nb in all_nbs[site_num]:
            if type(nb) == tuple:
                points.append(nb[0].coords)
         #   else:                              ########## 
         #       points.append(nb.site.coords) #  has not attribute of site.coords

        ### converting list to numpy array
        points = np.asarray(points)

        ### using tess object cntr to compute voronoi
        cntr = compute_voronoi(points)

        ### Voronoi vertices
        ### the first position in points is the site, therefore '0'
        v = get_vertices(0, cntr)

        for i in range(len(v)):
            ints_list.append(['B', v[i].tolist()])

        ### Voronoi face centers
        f = get_facecentroid(0, cntr)

        for j in range(len(f)):
            ints_list.append(['C', f[j].tolist()])

        ### Voronoi edge centers
        e = get_edgecenter(0, cntr)

        for k in range(len(e)):
            ints_list.append(['N', e[k].tolist()])

    ### return list of list ['Atom type', [x,y,z]]
    return ints_list


##########################################################
def get_pos_in_prim_cell(bulk_structure, a):
    """ Function to to map positions onto the primitive cell

    Parameters
        prim = pylada primitive cell
        a = cartesian coordinates of position

    Returns
        a2 = cartesian coordinates, such that fractional coordination = [0,1)
    """
    
    a1 = np.array(a)
    inv_cell = np.linalg.inv(bulk_structure.lattice.matrix)

    frac_a1 = np.dot(a1, inv_cell)

    frac_a1 = frac_a1 - np.rint(frac_a1)

    a2 = np.dot(bulk_structure.lattice, frac_a1)

    return a2


def get_interstitials(bulk_structure):
    """
    Get symmetry inequivalent interstitials
    """
    all_interstitials = get_all_interstitials(bulk_structure)
    
    imp_structure = bulk_structure.copy()
    
    for element in all_interstitials: 
        name, pos = element 
        try: 
            imp_structure.append('X', pos, coords_are_cartesian=True, validate_proximity=True) 
        except ValueError:
            continue 

    SA = SpacegroupAnalyzer(imp_structure)
    
    sim_structure = SA.get_symmetrized_structure()
    
    frac_positions = []

    for group in sim_structure.equivalent_sites:
        if group[0].specie.symbol != 'X':
            continue
        else:
            frac_positions.append( group[0].frac_coords )
            continue
    
    return frac_positions
    
