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
Optimize = True

# Set <TransLimit> to initial maximum translational movement in Angstroms and 
# <RhoLimitDeg> to initial maximum rotation in degres. If Optimize = True, both values
#  will be optimized togehre for a 50% acceptance rate but both will stay fixed otherwise:
TransLimit = 1
RhoLimitDeg = 30

# Set <ShepherdOn> to True in order to keep the centroid of the adsorbate within a 
# parralellepipid lattice with lattice vectors defined by the column 3-vectors in 
# <fence> and with its center at the centoid of the adsorbate.  If you don't want to
# take the time to define those vectors, just set this to false, but your adsorbate 
# might walk over to the adjacent unit cell (which might make data harder to analyze).
ShepherdOn = True

#####################################################################################
# Edit these to change the performance of the program: 
#####################################################################################
# Number of parallel MC processes you want to run (Only 1 is currently supported):
NumberOfProcesses = 1
# The number of cores to run SIESTA on:
NumberOfSiestaCores = 1