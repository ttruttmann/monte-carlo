#!/usr/bin/python
# Directives for the job manager:
#PBS -q default
#PBS -o output/output.pbs
#PBS -e output/error.pbs
#PBS -l walltime=1000:00:00
#PBS -l nodes=1:ppn=16

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
from MC_Util import attach # Used to initially place the admolecule onto the adsorbent
from MC_Util import Logger # Used for logging performance and writing to time.log.
from MC_Util import SIESTA # Used for calling SIESTA and retreiving energy for the baseline.
from MC_Util import AdEnergy # Used to calculate the binding energy.
from MC_Util import MC_Cycle # The "heart" of the MC routine.

# These are setting variables defined by the user in MC_Pref.py:
try: from MC_Pref import NumberOfSiestaCores
except: from MC_Defaults import NumberOfSiestaCores
try: from MC_Pref import NumberOfProcesses
except: from MC_Defaults import NumberOfProcesses
try: from MC_Pref import SubLatticeGrid
except: from MC_Defaults import SubLatticeGrid
try: from MC_Pref import AdmoleculeSize
except: from MC_Defaults import AdmoleculeSize
try: from MC_Pref import CounterpoiseOn
except: from MC_Defaults import CounterpoiseOn
try: from MC_Pref import RightSide
except: from MC_Pref import RightSide

# Since counterpoise calculations are not supported yet, I have to raise an error:
if CounterpoiseOn:
    sys.stderr.write('Sorry, counterpoise calulations are not supported!')
    raise Exception

# This is to begin measuring the duration of the program (more Loggers will be created
# in each process):
ProgramLogger = Logger() # This defines a Logger for the main process.
ProgramLogger.process.start() # This times the time spent actually running this process.
# This times the from the beginning to the end of the program (for the sake of parallell performance):
ProgramLogger.SuperProcess.start()

###
# This is a script that uses SIESTA potential energy calculations to run a
# Monte Carlo simulation.  It is designed to model any interaction between a single
# admolecule molecule and its adsorbant by freezing the admolecule and adsorbant, but # Note: After I add optomization, change this descrption.
# allowing the admolecule to translate and rotate. The default system is a 273-atom
# TNT-cellulose adorbtion process with effective 2-D periodic boundary conditions
# (SIESTA requires all 3 directions, so one is set very large), but can be
# adapted to model any single absorbate molecule adsorbtion process.  It is
# furthermore designed to run parallel processes.  This program will write output
# files to the folder that qsub or ./RunMC.py was called in.
# -Tristan Truttmann (tristan.truttmann@gmail.com)
# Note: Some things that I may do:
# Note: I need to spell check my comments
# Note: update estimates for the number of lines in comments.

# In the folowing ~20 lines it moves any old output files to an "archive_output###",
# folder, where ### is an index, to keep track of previous prefrences and their results.
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

# If we are going to do counterpoise calculations, we will record the basis-set-
# superposition energies in a file names SuperError.csv:
if CounterpoiseOn:
	buffer = '# This file record basis-set-superposition errors (corrected - uncorrected)'
	ProgramLogger.ReadWrite.start()
	file = open(CommandHome + '/SuperError.csv','w')
	file.write(buffer)
	file.close()
	ProgramLogger.ReadWrite.stop()

# Here I am copying the preferences file to the output so  that its easier to match
# results with prefrences.
shutil.copyfile(CommandHome + '/MC_Pref.py',CommandHome + '/output/MC_Pref[archive].py')
shutil.copytree(CommandHome + '/SiestaFiles', CommandHome + '/output/SiestaFiles[archive]')
# This reads from xyz file(s) (in SiestaFiles directory) to get the input structure
# and calculates the energy:
# There are two ways to designate the structure. We need to figure out which one first:
StructureFile = os.path.isfile(CommandHome + '/SiestaFiles/InputGeom.xyz')
AdsorbentFile = os.path.isfile(CommandHome + '/SiestaFiles/Adsorbent.xyz')
AdmoleculeFile = os.path.isfile(CommandHome + '/SiestaFiles/Admolecule.xyz')
# If the user supplis both types of files, then an error is raised:
if StructureFile and AdsorbentFile and AdmoleculeFile:
    message = 'You must supply either the adsorbent and admolecule in one file or seperate. '
    message += 'You may not supply both.'
    sys.stderr.wrie(message)
    raise Exception
# Here is the start-up if both are supplied in one structure.
if StructureFile:
    StartConfig = ConfigClass(CommandHome + '/SiestaFiles/InputGeom.xyz')
    if AdmoleculeSize == None:
        message = 'If you supply the adsorbent and admolecule together,'
        message += ' you must specify AdmoleculeSize.'
        sys.stderr.write(message)
        raise Exception
    StartConfig.AdmoleculeSize = AdmoleculeSize
# Here is the startup if the two are supplied in seperate xyz files.
elif AdsorbentFile and AdmoleculeFile:
    adsorbent = ConfigClass(CommandHome + '/SiestaFiles/adsorbent.xyz')
    admolecule = ConfigClass(CommandHome + '/SiestaFiles/admolecule.xyz')
    try: StartConfig = attach(adsorbent,admolecule,RightSide)
    except NameError:
        message = "You must have the statsmodels package in order to have the program"
        message += " automatically attach the admolecule to the adsorbent"
        sys.stderr.write(message)
        raise
StartConfig.QueryLattice(CommandHome + '/SiestaFiles/template.fdf')
StartConfig.SubLatticeGrid = SubLatticeGrid
baseline  = SIESTA(StartConfig.adsorbent(),CommandHome,NumberOfSiestaCores,ProgramLogger,ghost=None)
baseline += SIESTA(StartConfig.admolecule(),CommandHome,NumberOfSiestaCores,ProgramLogger,ghost=None)
StartConfig.E = AdEnergy(StartConfig,CommandHome,NumberOfSiestaCores,ProgramLogger,CounterpoiseOn,baseline,lock = multiprocessing.Lock())
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
    proc.append(multiprocessing.Process(target=MC_Cycle, args = (StartConfig, EnsembleSize, CommandHome, ScratchPath, WriteLock, LogQ,baseline,)))
    proc[i].start()

# Then we stop timing this process since it will soon sleep:
ProgramLogger.process.stop()
# This loop waits until each process sends it loggger infomation upon completion:
for i in range(NumberOfProcesses):
    ProgramLogger += LogQ.get()

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
ProgramLogger.summary(CommandHome + '/output/time.log')
