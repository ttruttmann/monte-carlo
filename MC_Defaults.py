# -*- coding: utf-8 -*-
import numpy as np

#####################################################################################
# A defaults file.  If the MC_Pref.py file does not have variables that the algorithm
# is seraching for, it will look here.  If the preferences are spelled wrong, it will
# look here.  
#####################################################################################

#####################################################################################
# Variables specific to systems: 
#####################################################################################
SubLatticeGrid = np.array((1,1,1))

# Enter the number of atoms of the admolecule.
AdmoleculeSize = None

# Enter a 3-element numpy array that points in the general direction of side of the 
# adsorbent that you want to attach to.  If you don't care, set it to None. 
RightSide = None
#####################################################################################
# Simulation parameters:
#####################################################################################
# Simulated anealing sequence, with the format...[[n_1,T_1],[n_2,T_2],...,[n_M,T_M]],
#  where n_m is the number of steps at T_m.  Each process will step through the 
# sequnce from left to right and restart when finished.  If you want constant 
# temperature, use the syntax [[1,T]], where T is your contant temperature.
T_Sequence = [[1,298]]

# Target number of ensemble images:
EnsembleTarget = 100

# Set to True if you want to dynamically optimize <TransLimit> and <RhoLimitDeg> or 
# set it to false for a fixed TransLimit and RhoLimitDeg.  True is recomended.
Optimize = False

# Set to True is you wish to perform a counterpoise calculation every time you perform
# an energyf caulculation.  Expected to increase load by 3x but recommended that you keep
# this option on: Sorry, counterpoise calculations are not supported yet. 
CounterpoiseOn = False

# Set <TransLimit> to initial maximum translational movement in Angstroms and 
# <RhoLimitDeg> to initial maximum rotation in degres. If Optimize = True, both values
#  will be optimized togehre for a 50% acceptance rate but both will stay fixed otherwise:
TransLimit = 1
RhoLimitDeg = 30

# Set <ShepherdOn> to True in order to keep the centroid of the admolecule within a 
# parralellepipid lattice with lattice vectors defined by the column 3-vectors in 
# <fence> and with its center at the centoid of the admolecule.  If you don't want to
# take the time to define those vectors, just set this to false, but your admolecule 
# might walk over to the adjacent unit cell (which might make data harder to analyze).
ShepherdOn = True

# <MetaDynamicsOn> will turn the MetaDynamics feature on or off. <MetaWidths> will specify
# the standard deviation of the Gaussian curves in Angstrom; the first three elements 
# are for x,y, and z and are in Angstrom.  The last 3 elements are for alpha, beta, and 
# gamma proper euler angles and are in degrees.
# <MetaHeights> will specify the hight of the curves in eV.  For course-grained 
# sampling, use larger hights and widths.  For fine-grained sampling, use smaller values. 
# If you wish to turn off local elevation for a certain dimension, set the width to 
# float('inf'). 
# Note: Think about how to specify height or area and then copy this to Pref file. 
# Also think about How big these default values should be.  
MetaDynamicsOn = True
MetaWidths = np.array((0.3,0.3,0.3,90,90,90))
MetaHeight = 0.05

#####################################################################################
# Edit these to change the performance of the program: 
#####################################################################################
# Number of parallel MC processes you want to run (Only 1 is currently supported):
NumberOfProcesses = 1
# The number of cores to run SIESTA on:
NumberOfSiestaCores = 16