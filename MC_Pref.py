# -*- coding: utf-8 -*-

import numpy as np

#####################################################################################
# Dear User: Please edit any of these variables to talor the algorithm to your specific 
# applicaiton.  This is a python script that the MC program will import variables from;
# therefore, you can use any python tricks that you already know. If you don't know
# python, read my instructions carefully, which assume you have very little python background. 
# If you comment out or misspell any of these variables, the defaults in MC_Defaults.py will be used. 
#####################################################################################

#####################################################################################
# Edit these variables to your specific system: 
#####################################################################################
# Enter the number of atoms of the admolecule.
AdmoleculeSize = 21  

# Please change these fence vectors to the smallest repeating unit in
# your adsorbant (not necessarily the unit cells entered in SIESTA). Please enter
# in [x,y,z] format.  If this takes too long and you would not like to fence in your 
# little admolecule, then disable this function by setting <ShepherdOn> to False. And 
# python won't even look for these variables.
SubLatticeGrid = np.array((2,3,1))
#####################################################################################
# Edit these to modify simulation parameters for your desired outcome:
#####################################################################################
# Dear User: This is a simulated anealing sequence, with the format
# [[n_1,T_1],[n_2,T_2],...,[n_M,T_M]], where n_m is the number of steps at
# T_m.  Each process will step through the sequnce from left to right and
# restart when finished.  If you want constant temperature, use the syntax
# [[1,T]], where T is your contant temperature.
T_Sequence = [[1,298]]

# Dear User: Enter desired number of ensemble images here:
EnsembleTarget = 10000

# Set to True if you want to dynamically optimize <TransLimit> and <RhoLimitDeg> or 
# set it to false for a fixed TransLimit and RhoLimitDeg.  True is recomended.
Optimize = False

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
# Note: Think about how to specify height or area and then copy this to Pref file. 
# Also think about How big these default values should be.  
MetaDynamicsOn = True
MetaWidths = np.array((0.3,0.3,0.3,90,90,90))
MetaHeight = 0.05

#####################################################################################
# Edit these to change the performance of the program: 
#####################################################################################
# Dear User: Enter the number of cores you would like to run SIESTA on...
# ...Also change the PBS directives at the beginning of RunMC.py accordingly:
NumberOfSiestaCores = 16

#Note: think about actually definiting the PBS directives here and then running this file, 
# which will run the RunMC file.  But maybe I'm just crazy. 