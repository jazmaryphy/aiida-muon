# -*- coding: utf-8 -*-

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



##########################################################
def get_cell_matrix( a, b, c, alpha, beta, gamma ):
    """ Generate a matrix for the lattice parameter
    Parameters:
    -----------
              a: lattice parameter in a direction angstrom
              b: lattice parameter in b direction angstrom
              c: lattice parameter in b direction angstrom
              alpha: angle between a and b
              beta:  angle between  b and c
              gamma: angle between c and a
    Return:
    -------
           A 3x3 matrix
    """
    alpha *= np.pi / 180
    beta *= np.pi / 180
    gamma *= np.pi / 180
    a1 = a
    a2 = 0.0
    a3 = 0.0
    b1 = np.cos( gamma )
    b2 = np.sin( gamma )
    b3 = 0.0
    c1 = np.cos( beta )
    c2 = ( 2 * np.cos( alpha ) + b1**2 + b2**2 - 2 * b1 * c1 - 1 ) / ( 2 * b2 )
    c3 = np.sqrt( 1 - c1**2 - c2**2 )
    matrix = np.zeros( ( 3, 3 ), dtype=float )
    matrix[ 0, 0 ] = a
    matrix[ 1 ] = np.array( [ b1, b2, b3 ] ) * b
    matrix[ 2 ] = np.array( [ c1, c2, c3 ] ) * c
    return matrix



##########################################################
def generate_uniform_grid(structure, size=3, min_distance_from_atoms=1.0):
    """
    Generates a grid of symmetry inequivalent interstitial
    positions with a specified minimum distance from the atoms of the
    sample. Especially intended for DFT simulations.

    :param bulk_structure: A structre data.
    :param int size: The number of steps in the three lattice directions.
                     Only equispaced grids are supported at the moment.
    :param float min_distance_from_atoms: Minimum distance between a
                                          interstitial position and the
                                          atoms of the lattice.
                                          Units are Angstrom.
    :returns: A list of symmetry inequivalent positions.
    :rtype: list
    """
    #bulk_structure=Structure.from_file(bulk_structure)
    bulk_structure = structure.copy()
    tolerance = 7

    #build uniform grid
    npoints = size**3
    x_ = np.linspace(0., 1., size, endpoint=False)
    y_ = np.linspace(0., 1., size, endpoint=False)
    z_ = np.linspace(0., 1., size, endpoint=False)

    uniform_grid = np.meshgrid(x_, y_, z_, indexing='ij')
    x,y,z = uniform_grid


    equiv=np.ones_like(x)*(npoints)

    SA = SpacegroupAnalyzer(bulk_structure)
    rot, tran = SA._get_symmetry()

    alt = True
    nb_cells = np.array([[-1., -1., -1.],
                         [-1., -1.,  0.],
                         [-1., -1.,  1.],
                         [-1.,  0., -1.],
                         [-1.,  0.,  0.],
                         [-1.,  0.,  1.],
                         [-1.,  1., -1.],
                         [-1.,  1.,  0.],
                         [-1.,  1.,  1.],
                         [ 0., -1., -1.],
                         [ 0., -1.,  0.],
                         [ 0., -1.,  1.],
                         [ 0.,  0., -1.],
                         [ 0.,  0.,  0.],
                         [ 0.,  0.,  1.],
                         [ 0.,  1., -1.],
                         [ 0.,  1.,  0.],
                         [ 0.,  1.,  1.],
                         [ 1., -1., -1.],
                         [ 1., -1.,  0.],
                         [ 1., -1.,  1.],
                         [ 1.,  0., -1.],
                         [ 1.,  0.,  0.],
                         [ 1.,  0.,  1.],
                         [ 1.,  1., -1.],
                         [ 1.,  1.,  0.],
                         [ 1.,  1.,  1.]])


    for i in range(size):
        for j in range(size):
            for k in range(size):

                if equiv[i,j,k] < npoints:
                    #this point is equivalent to someone else!
                    continue

                for r,t in zip(rot,tran):
                    # new position for the muon
                    n = np.zeros(3)
                    # apply symmetry and bring back to unit cell
                    n = np.round(np.dot(r,[x[i,j,k],y[i,j,k],z[i,j,k]])+t,decimals=tolerance)%1
                    if (np.abs(n*size - np.rint(n*size)) < 10**-(tolerance)).all():

                        #get index of point
                        ii,jj,kk=np.rint(n*size).astype(int)
                        if (ii*(size**2)+jj*size+kk > i*(size**2)+j*size+k):
                            equiv[ii,jj,kk] -= 1
                            equiv[i,j,k] += 1

    reduced_lat=np.array(bulk_structure.lattice)
    scaled_pos = bulk_structure.frac_coords

    A, B, C = structure.lattice.abc[0], structure.lattice.abc[1], structure.lattice.abc[2]
    alpha, beta, gamma = structure.lattice.angles[0], structure.lattice.angles[1], structure.lattice.angles[2]

    reduced_bases=get_cell_matrix( A, B, C, alpha, beta, gamma )


    positions = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if equiv[i,j,k] >= npoints:
                    #saves distances with all atoms
                    distances = []
                    dists = np.zeros(27*len(bulk_structure.sites))
                    center = [x[i,j,k],y[i,j,k],z[i,j,k]]

                    #check distances form atoms (also in neighbouring cells)


                    for a in range(len(bulk_structure)):
                        dists[a*27:(a+1)*27] = np.linalg.norm(
                                                  np.dot(scaled_pos[a] - center + nb_cells,
                                                         reduced_bases), axis=1 )
                    #
                    # old method 20 times slower!
                    #
                    #for a in range(len(sample._cell)):
                    #    for ii in (-1, 0, 1):
                    #        for jj in (-1, 0, 1):
                    #            for kk in (-1, 0, 1):
                    #                distances.append( np.linalg.norm(
                    #                        np.dot(scaled_pos[a] - center + np.array([ii,jj,kk]),
                    #                            reduced_bases) ) )
                    #
                    #print(np.allclose(dists,distances))
                    #if min(distances) > min_distance_from_atoms:


                    if dists.min() > min_distance_from_atoms:
                        positions.append([x[i,j,k],y[i,j,k],z[i,j,k]])

    return positions
