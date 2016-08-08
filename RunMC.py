#!/usr/bin/python

# Directives for the job manager:
#PBS -q default
#PBS -o output/output.pbs
#PBS -e output/error.pbs
#PBS -l walltime=1000:00:00
#PBS -l nodes=1:ppn=1

# The followign ~30 lines define imporant files and directories to write to and get from.
# It also extends the path for dependencies and imports a few variables from MC_Pref.py.
#  These will work as long as you call qsub or ./RunMC.py from the directory that contains
#  this script (and as long as the the other imporant dependencies are in that directory of course):

import multiprocessing
import os
import shutil
import sys
# <ScratchPath> is the path to the scratch directory.
if os.getenv("TMPDIR") != None:
   ScratchPath = os.getenv('TMPDIR') 
else:
   ScratchPath = os.getcwd()
# <CommandHome> is the path to where qsub or ./RunMC.py was called.
if os.getenv('PBS_O_WORKDIR') != None:
    CommandHome = os.getenv('PBS_O_WORKDIR')  
else:
    CommandHome = os.getcwd() 
sys.path.append(CommandHome)

# These are my own defined Classes that will be useful:
from MC_Util import ConfigClass
from MC_Util import Logger # Used for logging performance and writing to time.log.
from MC_Util import SIESTA # Used for calling SIESTA and retreiving energy.
from MC_Util import MC_Cycle # The "heart" of the MC routine.

# These are setting variables defined by the user in MC_Pref.py:
try: from MC_Pref import NumberOfSiestaCores
except: from MC_Defaults import NumberOfSiestaCores
try: from MC_Pref import NumberOfProcesses
except: from MC_Defaults import NumberOfProcesses
try: from MC_Pref import SubLatticeGrid
except: from MC_Defaults import SubLatticeGrid
try: from MC_Pref import AdsorbateSize
except: from MC_Defaults import AdsorbateSize

# This is to begin measuring the duration of the program (more Loggers will be created
# in each process):
ProgramLogger = Logger() # This defines a Logger for the main process. 
ProgramLogger.process.start() # This times the time spent actually running this process.
# This times the from the beginning to the end of the program (for the sake of parallell performance):
ProgramLogger.SuperProcess.start()

###
# This is a script that uses SIESTA potential energy calculations to run a
# Monte Carlo simulation.  It is designed to model any interaction between a single
# adsorbate molecule and its adsorbant by freezing the adsorbate and adsorbant, but # Note: After I add optomization, change this descrption.
# allowing the adsorbate to translate and rotate. The default system is a 273-atom
# TNT-cellulose adorbtion process with effective 2-D periodic boundary conditions
# (SIESTA requires all 3 directions, so one is set very large), but can be
# adapted to model any single absorbate molecule adsorbtion process.  It is
# furthermore designed to run parallel processes.  This program will write output
# files to the folder that qsub or ./RunMC.py was called in.
# -Tristan Truttmann (tristan.truttmann@gmail.com)
# Note: Consider adding some data analysis somehow.
# Note: Add umbrella sampling
# Note: Fix portability problem with my propensemble file. 
# Note: Make the sheppard function more robust
# Note: Add some more output files that store more infmation that can be analyzed by python. 
# Note: Read through Gopinath's code and adjust to his syntax. Also consider getting his newest edition.
# Note: look at how you start processes with a specific amount of resources.
# Note: check for the possibility of deadlocks.
# Note: Review notes and make sure all DONT follow the ...  and Dear User convention 
# Note: Think about making a defaults script for the backend.  Also think about hiding some of these backend scripts. 
# Note: Consider changing the permissions on the archive files I write. 
# Note: Think of other ways to decrase writing and reading overhead by skipping past N lines if there are N atoms etc.
# Note: Some things that I may do: 
# Note: I think I can remove Gopinath's routines from the code now.
# Note: I need to spell check my comments
# Idea: Create a fourier representation of the metdynamics bias to work with unitcells
# Idea: Create a datastructure that stores connectivity and maximum dihedral, angle, and bond perterpations for more efficient MC sampling. 
# Note: Make sure that for each thing in the ensemble file there is also one in all the ensemble files (no offset)
# Note: Make sure I test and verify this both in PBS, from command line, and maybe on Albacore. 
# Note: update estimates for the number of liens in comments.
# Note: Add an option whether to save archives or just delete them. 
##      -Add automatic calculation of adsorbate and andsorbant by themselves and then graph the adsorbtion energy (save the output files)
##      -Maybe create my own config class

# In the folowing ~20 lines it moves any old output files to an "archive_output###",
# folder, where ### is an index, to keep track of previous prefrences and their results.
# Note: The program still isn't terminating properly on errors.
# If there is an output folder with any files in it, we rename it ouput###, where ### 
# is the lowest 3-digit number which hasn't been used yet.  This will overwrite the 
# "output" directory if 1000 "archive_output###" directories already exist.  
if os.path.isdir(CommandHome + '/output'): # Checking of "output" directory exists
    if bool(os.listdir(CommandHome + '/output')): # If "output" exists, checking if it has files
        for i in range(1000):
            CandFolderName = CommandHome + '/archive_output' + str(i).zfill(3)
            # Checking if the "archive_output###" directory exists for this ### in the loop:
            if not os.path.isdir(CandFolderName): 
                # Moves "output" directory to "archive_output###" directory:
                shutil.move(CommandHome + '/output',CandFolderName)
                os.mkdir(CommandHome + '/output') # Remakes the "output" diretory.
                break
else:
    os.mkdir(CommandHome + '/output') # If the "output" directory didn't exists, it makes it.

# Here I am copying the preferences file to the output so  that its easier to match
# results with prefrences.
shutil.copyfile(CommandHome + '/MC_Pref.py',CommandHome + '/output/MC_Pref[archive].py')
shutil.copytree(CommandHome + '/SiestaFiles', CommandHome + '/output/SiestaFiles[archive]')
#try: # Note: I think I should use try earlier to prevent import eros from preferences file
    # This reads from InputGeom.xyz (in SiestaFiles directory) to get the input structure
    # and calculates the energy: # Note: I think I can delete the file objects
StartConfig = ConfigClass(CommandHome + '/SiestaFiles/InputGeom.xyz')
StartConfig.QueryLattice(CommandHome + '/SiestaFiles/template.fdf') 
StartConfig.SubLatticeGrid = SubLatticeGrid
StartConfig.AdsorbateSize = AdsorbateSize
SIESTA(StartConfig,CommandHome,NumberOfSiestaCores,ProgramLogger)
WriteLock = multiprocessing.Lock() # A lock for writing to all files.
LogQ = multiprocessing.Queue() # A queue for each process to send their Loggers to when finished.

# Then it writes the beginning of the poroperties ensemble file. #Note: I should time this writing process! 
StartConfig.StartWrite(CommandHome, lock = WriteLock)
StartConfig.Write(CommandHome, lock = WriteLock)

# This creates a variable that can be accesesed by different nodes for the number of
# structures written to the ensemble file:
EnsembleSize = multiprocessing.Manager().Value('i',0)

# The following ~15 lines start off <NumberOfProcesses> processes and then sleeps until 
# it receives logger informaiton in the queue (which is send as a Logger() instance
# when each of the subprocesses complete): 

# The first loop starts the processes:
proc = []
for i in range(NumberOfProcesses):
    proc.append(multiprocessing.Process(target=MC_Cycle, args = (StartConfig, EnsembleSize, CommandHome, ScratchPath, WriteLock, LogQ,)))
    proc[i].start()

# Then we stop timing this process since it will soon sleep:
ProgramLogger.process.stop()
# This loop waits until each process sends it loggger infomation upon completion:
for i in range(NumberOfProcesses):
    ProgramLogger += LogQ.get()

#finally: # Note: Finad a way to impliment his finally without getting locked.  

# Then we have to start timing this process again. 
ProgramLogger.process.start()

# Then the folliwing ~10 lines writes a final performacne summary to the end of the 
# time.log file, which is the end of the program:

# First we write a note that seperates is from other notes written from individual processes:
ProgramLogger.ReadWrite.start()
with open(CommandHome + '/output/time.log','w') as file:
    file.write('This is the performance information of the entire MC program.\n.')
    file.close()
ProgramLogger.ReadWrite.stop()

# Then we end all processes and write the actual performance summary:
ProgramLogger.process.stop()
ProgramLogger.SuperProcess.stop()
ProgramLogger.summary(CommandHome + '/output/time.log') # Note: I shold consider creating a logpath variable or something.  I should also update this function (have default variables.)