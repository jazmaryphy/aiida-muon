import os,sys, threading
import argparse

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

import logging as log
import numpy as np


"""import from muesr  for calculations"""

from muesr.core.sample import Sample
from muesr.i_o.xsf.xsf import load_xsf
from muesr.utilities.visualize import show_structure
from muesr.i_o.cif.cif import load_cif, load_mcif, load_mcif_file
from muesr.i_o.sampleIO import load_sample, save_sample
from muesr.core.sampleErrors import CellError, MuonError, MagDefError
from muesr.i_o.exportFPS import export_fpstudio
from muesr.engines.clfc import locfield, find_largest_sphere
from muesr.utilities import muon_find_equiv
from muesr.utilities.dft_grid import build_uniform_grid

def samplecall():
    """Initialize the Sample for the calculations """
    return Sample()

def _load_sample(filename):
    """Load the sample
    Parameters
    ----------
    filename : str
               the name of the MCIF, CIF file
    Returns
    -------
    None
    """
    if filename:
       try:
          return load_sample(filename)
       except:
           log.error("Could not load sample.", exc_info=True)
           return
    else:
         return

def load_structure_file(filename, sample):
    """Load the structure file
    Parameters
    ----------
    filename : str
               the name of the MCIF, CIF, XSF file
    sample:  
          load the sample environment for muesr
    Returns
    -------
    None
    """
    if not filename:
       return
       
    froot,fext = os.path.splitext(filename)
    if fext.lower() == '.xsf':
       load_xsf(sample,filename)
    elif fext.lower() == '.cif':
       load_cif(sample,filename)
    elif fext.lower() == '.mcif':
       load_mcif_file(sample,filename)
    else:
        log.error("Invalid file extension for file {}.".format(froot))
    return

def show_cell(structure):
    if structure:
       try:
          s=samplecall()
          load_structure_file(structure, s)
          show_structure(s , [1,1,1], )
       except CellError:
           log.error("Lattice structure not defined!")


def generate_uniform_grid(filename, grid_size):
    """Generate a uniform grid of points
    Parameters
    ----------
    filename : str
                the name of the MCIF, CIF file
    grid_size : int
                a grid dimension equal for all direction
    Returns
    ------
    an array of inital position in fractional coordinates
    """
    s=samplecall()
    load_structure_file(filename, s)
    grid_pos=build_uniform_grid(s, grid_size)
    grid_pos=np.array(grid_pos)
    return  grid_pos

def show_structure_with_muon(filename, mu_position,  sc_size=(1,1,1), cartesian=False):
    """show the structure with muon using XCRYSDEN Software
    Parameters
    ----------
    filename : str
               the name of the MCIF, CIF file   
    mu_position: 
               the muon position to add to the bulk structure
    sc_size :
             A supercell dimension to enlarge the unitcell. Default=(1,1,1)
    Returns
    -------
    a GUI Xcrysden.  
    """
    if filename:
       try:
          s=samplecall()
          load_structure_file(filename, s)
          s.add_muon(list(mu_position),  cartesian=cartesian)
          show_structure(s , list(sc_size))
       except CellError:
           log.error("Lattice structure or Muon not defined!")

