###
# Utilities used in RunMC.py for its Monte Carlo algorithm.
# -Tristan Truttmann (tristan.truttmann@gmail.com)
###

# The followign ~20 lines define imporant files and directories to write to and get from.
# It also extends the path for dependencies:

# These are python libraries that are needed:
import copy
import os
import multiprocessing
import numpy as np
import shutil
import sys
import time

# These are paths that are needed:
# <CommandHome> is the path to where qsub or ./RunMC.py was called:
# If we are running in PBS, use the PBS working directory environment variable for <CommandHome>:
if os.getenv('PBS_O_WORKDIR') != None:
    CommandHome = os.getenv('PBS_O_WORKDIR')
# Otherwise, just use the directory where the script was called:
else:
    CommandHome = os.getcwd()
# Then append paths that contain dependencies:
sys.path.append(CommandHome)

#====================================================================================
class PTableClass:
# Class that contains two methods:
## <anum(sym)> Accepts atomic symbol (string) and returns atomic number (int).
## <symbol(anum)> Accepts atomic number (int) and returns atomic symbol (string).
#====================================================================================
    def __init__(self):
        # A list object that stores atomic symbols.
        self.symbols = [
        "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
        "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
        "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb",
        "Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho",
        "Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi",
        "Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es",
        "Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Uut","Fl","Uup",
        "Lv","Uus","Uuo"]

    def anum(self,sym):
        try:
            class NoElement(Exception): pass
            for i in range(len(self.symbols)):
                if self.symbols[i] == sym:
                    return i + 1
            raise NoElement
        except NoElement:
            sys.stderr.write("There is no element with symbol " + str(sym) + ".\n")
            raise

    def symbol(self,anum):
        try:
            class NoElement(Exception): pass
            return self.symbols[anum - 1]
        except IndexError:
            sys.stderr.write("There is no element with atomic number " + str(anum) + ".\n")
            raise

# Then we create an instance to be imported.
ptable = PTableClass()

# The next ~5 lines define exceptions that are used:
# Raised when trying to read from an incompatible xyz format:
class FileFormatError(Exception): pass
# Raised when dimenstions of a ConfigClasses anum and coord arrays to not align:
class SizeInconsistency(Exception): pass



#====================================================================================
class ConfigClass:
# Inspired from Dr. Gopinath Subramanian's ConfigClass in molpro-neb/ClassDefinition.py,
# this class contains informaiton of atomic positions and identities, as well as
# characteristics such as energy and the centroid, as well as a few other methods that
# make life a lot easier during this Monte Carlo simulation.
# Attributes:
#   anum:               A numpy array of integers storing the atomic number of each atom.
#
#   coord:              A nx3 numpy array of loats storing the x, y, and z positions of each atom.
#
#   E:                  Storing the energy of the configuration in eV. Default 0.0
#
#   bias:               A float storing the selectino bias that was used to acheive the
#                       configuration in the MC. Default 1.0.
#
#   AdsorbateSize:      An integer storing the number of atoms in the adsorbate.
#
#   LatticeMatrix:      A 3x3 array storing the 3 lattice vectors of the whole system.
#                       Default None.
#
#   LatticeGrid:        A 3-element integer numpy aray that stores the number of repeating
#                       unites of the adsorbent inside on repeating unit of the entire system.
# Methods:
#   __init__():         Can be initializd with (1) zero arguments to create empty configuration,
#                       (2) with a string of an xyz file, or (3) with another ConfigClass instance
#                       which it will copy.
#
#   centroid():         Returns the centroid of the system as a 3-element numpy array.
#
#   adsorbent():        Returns another ConfigClass that only contains the atoms of the
#                       adsorbent. Only works if AdsorbateSize is properly defined.
#
#   adsorbate():        Returns another ConfigClass that only contains the atoms of the
#                       ad-molecule. Only works if AdsorbateSize is properly defined.
#
#   verify():           Checks to ensure that the sizes of anum and coord are consistent.
#                       Raises an error if they are inconsisstent.
#
#   WriteGeom():        Appends the geometry to an xyz file.  Like all write meathods,
#                       the first arguement is name of xyz file and the second (optional)
#                       argument is a lock for writing to files.
#
#   StartWriteEnergy(): Creates a python file to write the energy to and writes the
#                       Important information at the beginning of the file
#
#   WriteEnergy():      Appends energy information to the specied file in a python-
#                       readable format.
#
#   StartWriteConfig(): Creates a python file to write configuration information to.
#
#   WriteConfig():      Appends configuration information to the specified file in a
#                       python-readable format.  The goal is to be able to use this
#                       file to regenerate the ConfigClass object if necessary.
#
#   StartWrite():       Runs StartWriteEnergy() and StartWriteConfig(). First argument
#                       is the directory that the files will be written to, and the
#                       second (optional) argument is a lock for writing to files.
#                       It will generte files names named EnergyEnsemble.py and ConfigEnsemble.py
#                       and will overwrite existing files.
#
#   Write():            Runs WriteGeom(), WriteEnergy(), and WriteConfig(). First
#                       argument is the directory that the files will be written to,
#                       and the second (optional) argument is a lock for writing to
#                       files. Will append to files named GeomEnsemble.xyz,
#                       EnergyEnsemble.py, and ConfigEnsemble.py.
#
# Note: I need to later make a function to get the lattice infomraiton from the file.
#====================================================================================
    def __init__(self,father=None):
        # If there is no argument, an empty default ConfigClass is created:
        if father is None:
            self.anum = np.zeros(0 , dtype = int)
            self.coord = np.zeros((0,3), dtype=float)
            self.E = 0.0
            self.bias = 1.0
            self.AdsorbateSize = None
            self.LatticeMatrix = None
            self.SubLatticeGrid = None # Note: Make sure I'm using these things!
        # If the argument is a string, then it attempts to read it as an xyz file:
        elif type(father) is str:
            try:
                # First some defaults are set that are not available from the xyz file:
                self.E = 0.0
                self.bias = 1.0
                self.AdsorbateSize = None
                self.LatticeMatrix = None
                self.SubLatticeGrid = None # Note: Change shepperd to use these!
                # It opens the file and reads the number of atoms from line 1:
                file =  open(father,'r')
                natom = int(file.readline())
                # It creates empty arrays for soon-to-come structural information:
                self.anum = np.zeros(natom , dtype = int)
                self.coord = np.empty((natom,3),dtype=float)
                # Then it skips past the title line:
                file.readline()
                # Then it steps through every atom in the file
                # (accodring to the first line):
                for iatom in range(natom):
                    # I parses the line and converts symbols to atomic numbers in
                    # the anum array:
                    atom = file.readline().split()
                    self.anum[iatom] = ptable.anum(atom[0])
                    # Then it steps through every coordinate in the line and records
                    # them in the coord array:
                    for icoord in range(3):
                        self.coord[iatom][icoord] = float(atom[icoord + 1])
                # Then it check to make sure there is no other information in the file
                # to check for errors.  It raises an error if necessary:
                nextline = file.readline().replace(' ','').replace('\t','').replace('\n','')
                if nextline != '':
                    raise FileFormatError
                file.close()
            # It handels three likely erros with a short error message and raises
            # the error again:
            except(ValueError,IndexError,FileFormatError):
                sys.stderr.write('The file ' + father + ' either has an unsupported format or has several structures.  Delete extra structures or fix formating error.\n')
                raise
        # If there is an argument, but it is not a string, it will assume that
        # it is another ConfigClass instance:
        else:
            try:
                # It first runs the verify method, which will raise an error if it
                # finds father to be invalid:
                father.verify()
                # Then it creates a deep copy and assigns self's attributes to the copy's:
                CopyConfig = copy.deepcopy(father)
                self.anum = CopyConfig.anum
                self.coord = CopyConfig.coord
                self.E = CopyConfig.E
                self.AdsorbateSize = CopyConfig.AdsorbateSize
                self.bias = CopyConfig.bias
                self.LatticeMatrix = CopyConfig.LatticeMatrix
                self.SubLatticeGrid = CopyConfig.SubLatticeGrid
            # Then it handles the error by writing to standard error and raising again:
            except (AttributeError,SizeInconsistency):
                sys.stderr.write( "You have tried to initialize a ConfigClass with some incompatiblae Object. Perhaps you used Dr. Subramanian's ConfigClass in molpro-neb/ClassDefinition.py.  Consider using the one from MC_Util.py." )
                raise
        self.verify()
        
    # A method to retreive lattice information from a fdf file:
    def QueryLattice(self,FileName):
        SuccessMessage = 'The lattice information was successfully retreived.\n'
        # If no lattice information is found, LatticeMatrix defaults to None.
        self.LatticeMatrix = None
        # Then it opens the file and reads through until it finds lattice
        # information keywords:
        file = open(FileName,'r')
        while True:
            line = file.readline()
            if line == '': 
                print 'The lattice information could not be found. LatticeMatrix = None\n'
                break
            # If the fdf file provides lattice vectors, getting the vectors is quite easy:
            if '%block latticevectors' in line.lower():
                # First it creates the matrix of zeros:
                self.LatticeMatrix = np.zeros((3,3))
                # Then each vector is read:
                for ivec in range(3):
                    # Each vector is split into a string list:
                    StringVec = file.readline().split()
                    for idim in range(3):
                        # Each element is interpreted as a float and assigned to matrix:
                        self.LatticeMatrix[idim,ivec] = float(StringVec[idim])
                # Then it raises an error if the end of the block is not found:
                if '%endblock latticevectors' not in file.readline().lower():
                    raise FileFormatError
                print SuccessMessage
                break
            # If the fdf file provides lattice parameters (crystalography format), 
            # then getting the lattice vectors is a little bit more involved:
            if '%block latticeparameters' in line.lower():
                line = file.readline().split()
                # First it parses the line and retrives lattice parameters:
                a = float(line[0])
                b = float(line[1])
                c = float(line[2])
                RadAlpha = float(line[3]) * np.pi / 180
                RadBeta = float(line[4]) * np.pi / 180
                RadGamma = float(line[5]) * np.pi / 180
                # Then it raises an error if the end of the block is not found:
                if '%endblock latticeparameters' not in file.readline().lower():
                    raise FileFormatError
                # Then it find each element of the matrix with given info:
                x1 = a
                y1 = 0
                z1 = 0
                x2 = b * np.cos(RadGamma)
                y2 = b * np.sin(RadGamma)
                z2 = 0
                x3 = c * np.cos(RadBeta)
                y3 = (b * c * np.cos(RadAlpha) - x2*x3) / y2
                z3 = np.sqrt(c**2 - x3**2 - y3**2)
                # Then it assigns the information to the LatticeMatrix:
                self.LatticeMatrix = np.array([(x1,x2,x3),(y1,y2,y3),(z1,z2,z3)])
                print SuccessMessage
                break
        return

    # The default representation prints the number of atoms:
    def __repr__(self):
        return "ConfigClass Object of " + str(len(self.anum)) + " atoms."
    
    # When converted to a string, it lists the atomic numbers of all atoms:
    def __str__(self):
        return self.anum.__str__()

    # Tool to return cetnroid of system. Adapted from Dr. Gopinath Subramanian's
    # molpro-neb routines:
    def centroid(self):
        return np.average(self.coord,axis=0)
        
    # Returns a new ConfigClass instance that represents the adsorbent. Only works if 
    # AdsorbateSize is properly defined:
    def adsorbent(self):
        # First it makes a copy of self and declares a new ConfigClass:
        TotalSystem = copy.deepcopy(self)
        NewConfig = ConfigClass()
        # Then is assigns values to attributes appropriately:
        NewConfig.anum = TotalSystem.anum[: len(TotalSystem.anum)-TotalSystem.AdsorbateSize]
        NewConfig.coord = TotalSystem.coord[: len(TotalSystem.coord)-TotalSystem.AdsorbateSize]
        NewConfig.E = 0.0
        NewConfig.bias = TotalSystem.bias
        NewConfig.AdsorbateSize = TotalSystem.AdsorbateSize
        NewConfig.LatticeMatrix = None
        NewConfig.SubLatticeGrid = None
        return NewConfig
        
    # Returns a new ConfigClass instance that represents the adsorbate. Only works 
    # if AdsorbateSize is properly defined:
    def adsorbate(self):
        TotalSystem = copy.deepcopy(self)
        NewConfig = ConfigClass()
        NewConfig.anum = TotalSystem.anum[len(TotalSystem.anum)-TotalSystem.AdsorbateSize :]
        NewConfig.coord = TotalSystem.coord[len(TotalSystem.coord)-TotalSystem.AdsorbateSize:]
        NewConfig.E = 0.0
        NewConfig.bias = TotalSystem.bias
        NewConfig.AdsorbateSize = TotalSystem.AdsorbateSize
        NewConfig.LatticeMatrix = None
        NewConfig.SubLatticeGrid = None
        return NewConfig

    # Raises an error if inconsistencies are found. It checks the 
    # datatype and length of the anum and coord arrays:
    def verify(self):
        try:
            valid = len(self.anum) == len(self.coord)
            valid = valid and type(self.anum) is type(self.coord) is np.ndarray
            if not valid:
                raise SizeInconsistency
        except SizeInconsistency:
            sys.stderr.write('There is a size inconsistency in your ConfigClass object.')
            raise

    # Writes geometry of self to xyz file:
    def WriteGeom(self,FileName,lock = multiprocessing.Lock()):
        # First check for size inconsistencies:
        self.verify()
        # Then write number of atoms:
        buffer = str(len(self.coord)) + '\n\n'
        # Then loop through every atom:
        for iatom in range(len(self.anum)):
            # Then convert the numbers in anum to the atomic symbols:
            buffer += ptable.symbol(self.anum[iatom])
            # Then loop through every coordinate in each atom:
            for icoord in range(3):
                buffer += ' ' + str(self.coord[iatom][icoord])
            buffer += '\n'
        # Then append the buffer to the file:
        lock.acquire()
        file = open(FileName,'a')
        file.write(buffer)
        file.close()
        lock.release()
        return

    # Prepares a file to write the energies to in a python-readable format:    
    def StartWriteEnergy(self,FileName,lock = multiprocessing.Lock()):
        # First it checks for size inconsistencies:
        self.verify()
        # Then it imports numpy:
        buffer = 'import numpy as np\n'
        # Then it initializes an empty numpy array:
        buffer += 'energy = np.array((),dtype=float\n'
        # Then it writes the buffer to the file:
        lock.acquire()
        file = open(FileName,'w')
        file.write(buffer)
        file.close()
        lock.release()
        return

    # Appends enrgy data to a python-readable file:    
    def WriteEnergy(self,FileName,lock = multiprocessing.Lock()):
        # First it checks for size inconsistencies:
        self.verify()
        # Then it directly writes the energy to the file:
        lock.acquire()
        file = open(FileName,'a')
        file.write('energy = np.append(energy,' + str(self.E) + ')\n')
        file.close()
        lock.release()
        return

    # Generates a new file to record the most important ConfigClass attributes to:    
    def StartWriteConfig(self,FileName, lock = multiprocessing.Lock()):
        # First it checks for size inconsistencies:        
        self.verify()
        # Then it imports necessary libraries and initializes an empty list:
        buffer = 'import sys\n'
        buffer += 'from numpy import *\n'
        buffer += 'ConfigList = []\n'
        buffer += '# Dear User: Append the directory where MC_Util.py is\n'
        buffer += 'sys.path.append("..") # (Assumes in directory above)\n'
        buffer += 'from MC_Util import ConfigClass\n\n'
        # Then it writes the buffer to the file:        
        lock.acquire()
        file = open(FileName,'w')
        file.write(buffer)
        file.close()
        lock.release()
        return

    # Appends most important ConfigClass attributes to a python-readable file:
    def WriteConfig(self, FileName, lock = multiprocessing.Lock()):
        # First it checks for size inconsistencies:
        self.verify()
        # Then it writes important information to buffer:
        buffer = '\nTempConfig = ConfigClass()\n'
        buffer += 'TempConfig.anum = ' + repr(self.anum) + '\n' #Note: Make sure this works!
        buffer += 'TempConfig.coord = ' + repr(self.coord) + '\n'
        buffer += 'TempConfig.E = ' + repr(self.E) + '\n'
        buffer += 'TempConfig.bias = ' + repr(self.bias) + '\n'
        buffer += 'TempConfig.AdsorbateSize = ' + repr(self.AdsorbateSize) + '\n'
        buffer += 'TempConfig.LatticeMatrix = ' + repr(self.LatticeMatrix) + '\n'
        buffer += 'TempConfig.SubLatticeGrid = ' + repr(self.SubLatticeGrid) + '\n'
        buffer += 'ConfigList.append(ConfigClass(TempConfig))\n'
        # Then it appends the buffer to the file:
        lock.acquire()
        file = open(FileName,'a')
        file.write(buffer)
        file.close()
        lock.release()
        return

    # Runs StartWriteEnergy() and StartWriteConfig() automatically:
    def StartWrite(self, CommandHome, lock = multiprocessing.Lock() ):
        try:
            # First it makes sure the path ends with '/':
            if not CommandHome.endswith('/'):
                CommandHome += '/'
            # Then it calls the two functions:
            self.StartWriteEnergy(CommandHome + '/output/EnergyEnsemble.py',lock = lock)
            self.StartWriteConfig(CommandHome + '/output/ConfigEnsemble.py',lock = lock)
            return
        except AttributeError:
             sys.stdout.write('The first argument of ConfigClass().StartWrite() must be a string\n')
             raise
             
    # Runs WriteGom, WriteEnergy, and WriteConfig automatically:
    def Write(self, CommandHome, lock = multiprocessing.Lock() ):
        try:
            # First it makes sure the path ends with '/':
            if not CommandHome.endswith('/'):
                CommandHome += '/'
            # Then it calls the three functions and returns:
            self.WriteGeom(CommandHome + '/output/GeomEnsemble.xyz', lock = lock)
            self.WriteEnergy(CommandHome + '/output/EnergyEnsemble.py', lock = lock)
            self.WriteConfig(CommandHome + '/output/ConfigEnsemble.py', lock = lock)
            return
        except AttributeError:
             sys.stdout.write('The first argument of ConfigClass().StartWrite() must be a string\n')
             raise

#========================================================================================================
def RotateAxisAngle(conf, unit_axis, DegAngle):
# A function adapted from Dr. Gopinath Subramanian's molpro-neb routines.
# Rotates the configuration about the origin, about unit_axis, by angle (degrees).
#========================================================================================================
    newconf = copy.deepcopy(conf)
    RadAngle = DegAngle * np.pi / 180    

    ct = np.cos(RadAngle)
    st = np.sin(RadAngle)
    u = unit_axis[0]
    v = unit_axis[1]
    w = unit_axis[2]
      
    x = copy.deepcopy(conf.coord[:,0])
    y = copy.deepcopy(conf.coord[:,1])
    z = copy.deepcopy(conf.coord[:,2])
    newconf.coord[:,0] = u*(u*x+v*y+w*z)*(1-ct) + x*ct + (-w*y+v*z)*st
    newconf.coord[:,1] = v*(u*x+v*y+w*z)*(1-ct) + y*ct + ( w*x-u*z)*st
    newconf.coord[:,2] = w*(u*x+v*y+w*z)*(1-ct) + z*ct + (-v*x+u*y)*st
    
    return newconf

#====================================================================================
def MC_Cycle(StartConfig, EnsembleSize, CommandHome, ScratchPath, WriteLock, LogQ):
# The "backbone" of the MC algorithm. Will be initated as many parallel processes 
# which will each create new ensemble images and add them to a shared ensemble record.
# The cycles will terminate when EnsembleTarget is reached.
#====================================================================================
    # First it imports settings chosen by the user:
    try:
        from MC_Pref import EnsembleTarget
    except ImportError:
        from MC_Defaults import EnsembleTarget
    try:
        from MC_Pref import NumberOfSiestaCores
    except ImportError:
        from MC_Defaults import NumberOfSiestaCores
    try:
        from MC_Pref import Optimize
    except ImportError:
        from MC_Defaults import Optimize
    try:
        from MC_Pref import RhoLimitDeg
    except ImportError:
        from MC_Defaults import RhoLimitDeg
    try:
        from MC_Pref import ShepherdOn
    except ImportError:
        from MC_Defaults import ShepherdOn
    try:
        from MC_Pref import TransLimit
    except ImportError:
        from MC_Defaults import TransLimit
    try:
        from MC_Pref import T_Sequence
    except:
        from MC_Defaults import T_Sequence

    CycleLogger = Logger()
    CycleLogger.process.start()
    # First the variables defined globally are declared in the funciton and scientific
    # constants are defined:

    k = 8.6173324e-5 # in eV K^(-1) from Wikipedia.
    # Then it begins the process of defining new structures and testing them:
    OldConfig = ConfigClass(StartConfig)
    T_index = [0,0]
    while True:
        # First it handles the simulated anealing sequence:\
        # Note: Consider making this an iterable function.
        T = T_Sequence[T_index[0]][1]
        T_index[1] += 1
        if T_index[1] == T_Sequence[T_index[0]][0]:
            T_index[1] = 0
            T_index[0] += 1
            if T_index[0] == len(T_Sequence):
                T_index[0] = 0
        CandConfig = move(OldConfig,TransLimit,RhoLimitDeg) # Note: Update this move function for new configClass.
        CandConfig = shepherd(CandConfig) if ShepherdOn else CandConfig
        SIESTA(CandConfig,CommandHome,NumberOfSiestaCores,CycleLogger)
        rand = np.random.random()
        prob = np.exp((OldConfig.E-CandConfig.E)/(k*T))
        if rand <= prob:
            CycleLogger.ReadWrite.start()
            CandConfig.Write(CommandHome, lock = WriteLock)
            CycleLogger.ReadWrite.stop() # Note: Consider making one big lock to preventthreading of writing processes.
            OldConfig = ConfigClass(CandConfig) # This is a way to assign without side-effects
            # Note: this is a big mess.  Fix all this some time.
            EnsembleSize.value = EnsembleSize.value + 1
            CycleLogger.hit()
            if Optimize:
                TransLimit = TransLimit * 1.05
                RhoLimitDeg = RhoLimitDeg * 1.05
            if EnsembleSize.value >= EnsembleTarget:
                CycleLogger.process.stop()
                LogQ.put(CycleLogger)
                break
        else:
            CycleLogger.miss()
            print "The TransLimit = " + str(TransLimit) # Note: Delete these. For debuggin purposes.
            print "The RhoLimitDeg = " + str(RhoLimitDeg)
            if Optimize:
                TransLimit = TransLimit * 0.95
                RhoLimitDeg = RhoLimitDeg * 0.95
        CycleLogger.ReadWrite.start()
        CycleLogger.summary(CommandHome + '/time.log',WriteLock)
        CycleLogger.ReadWrite.stop() # Note: I'm not yet sure if I should do this outside or inside the summary funciton.
    # CycleLogger.process.stop() Note: I think I have to delete this and add the queue transfer after this.

#====================================================================================
def SIESTA(config,CommandHome,NumberOfSiestaCores,Logger):
# This function uses the file template.fdf in the SiestaFiles directory as a template
# to run a SIESTA job to compute the potential energy.  It appends the geometric
# coordinates to the end of the file in the form "X\tY\tZ\tN" where N is the species
# number (which is extracted from template.fdf.  If you want to change the energy
# calculation parameters, edit the template.fdf file.
#====================================================================================
    # First the working directory <WorkDir> is created, and the pseudopotentials
    # <*.psf> and imput file <template.fdf> are coppied to it:
    WorkDir = '/tmp/' + os.getlogin()
    if not os.path.isdir(WorkDir):
        os.mkdir(WorkDir)
    WorkDir = WorkDir + '/proc' + str(multiprocessing.current_process().pid)
    if not os.path.isdir(WorkDir):
        os.mkdir(WorkDir)
    Logger.ReadWrite.start()
    shutil.copy(CommandHome + '/SiestaFiles/template.fdf', WorkDir + '/')
    os.system('cp ' + CommandHome + '/SiestaFiles/*.psf ' + WorkDir + '/' )
    Logger.ReadWrite.stop()
    # Then we have to define a species dict using the <template.fdf> file.  This will
    # convert between atomic symbols and Species Numbers that SIESTA uses:
    # Note:  This might be too costly to do this during every iteration...  I should move it to min script or somehting.
    Logger.ReadWrite.start()
    with open(WorkDir + '/template.fdf','r') as file:
        SpeciesNumbers = dict()
        NoteTaking = False
        for line in file:  # Note: Consider doing this somewhere else to save on reading time.
            if '%block ChemicalSpeciesLabel' in line:
                NoteTaking = True
                continue
            if '%endblock ChemicalSpeciesLabel' in line:
                break
            if NoteTaking == True:
                Note = line.replace('\n','').replace('\t','').replace(',','').split(' ')
                while '' in Note: # Note: Do I really need to have a while loop Here?
                    Note.remove('')
                SpeciesNumbers[Note[1]] = Note[0]
        file.close()
    Logger.ReadWrite.stop()
    # Then we define an output buffer <OutBuffer> in memory to store the string that
    # that we will write to the file:
    OutBuffer = '%block AtomicCoordinatesAndAtomicSpecies\n'
    for i in range(len(config.anum)):
        for j in range(3): # Note: This must be changed for new ConfigClass!
            OutBuffer += str(config.coord[i][j]) + '\t'
        OutBuffer += SpeciesNumbers[str(config.anum[i])]
        OutBuffer += '\n'
    OutBuffer = OutBuffer + '%endblock AtomicCoordinatesAndAtomicSpecies'
    # Then we write the geometry block to the file:
    Logger.ReadWrite.start()
    with open(WorkDir + '/template.fdf','a') as file:
       file.write(OutBuffer)
       file.close()
    Logger.ReadWrite.stop()
    # Then it moves into the <WorkDir> directory, runs the SIESTA job, then moves up:
    os.chdir(WorkDir)
    Logger.siesta.start()
    os.system('mpirun -np ' + str(NumberOfSiestaCores) + ' siesta < template.fdf | tee template.out')
    Logger.siesta.stop()
    os.chdir('..')
    #Then we extract the energy from the output file:
    Energy = 0.00
    # Dear User: This only works for harris functional.  If you edit <template.fdf>,
    # please replace 'siesta: Eharris(eV) =' appropriately.
    Logger.ReadWrite.start() # Note: Make some lookup routine to generalize energy mining.
    with open(WorkDir + '/template.out','r') as file:
        for line in file:
            if 'siesta: Eharris(eV) =' in line:
                line = line.replace('siesta: Eharris(eV) =','')
                Energy = float(line)
                break
        file.close()
    Logger.ReadWrite.stop()
    # Finally it deletes the working directory <WorkDir> and writes the Enegy in eV
    # to <config.E>:
    shutil.rmtree(WorkDir)
    config.E = Energy

#====================================================================================
def move(config,TransLimit,RhoLimitDeg):
# Returns a randomly pertubed configuration given an original configuration. This
# function works with any single molecule absorbate system as long as the adsorbate
# is the last atoms listed in the ConfigClass. Translates it over a flat distribution
# in any direction and rotates it over a flast angle distribution on a random axis.
#====================================================================================
    RhoLimitDeg = min(RhoLimitDeg,360)    
    CopyConfig = copy.deepcopy(config)
    ThetaTrans = np.random.rand() * 2 * np.pi
    CosPhiTrans = np.random.rand() * 2 - 1
    RandTrans = np.array((0.0,0.0,0.0))
    RandTrans[0] = np.cos(ThetaTrans) * np.sqrt(1-CosPhiTrans**2)
    RandTrans[1] = np.sin(ThetaTrans) * np.sqrt(1-CosPhiTrans**2)
    RandTrans[2] = CosPhiTrans
    RandTrans = RandTrans * np.random.rand() * TransLimit
    ThetaTurn = np.random.rand() * 2 * np.pi
    CosPhiTurn = np.random.rand() * 2 - 1
    RandAxis = np.array((0.0,0.0,0.0))
    RandAxis[0] = np.cos(ThetaTurn) * np.sqrt(1-CosPhiTurn**2)
    RandAxis[1] = np.sin(ThetaTurn) * np.sqrt(1-CosPhiTurn**2)
    RandAxis[2] = CosPhiTurn
    RandRho = (2 * np.random.rand() - 1) * RhoLimitDeg
    # Note: I just adjusted it so Translimit is absolute (it is now normalized)
    # Note: Here is a good place to get material for my honors!
    # Note: My convension is the rho is the rotation around r and phi and theta describe r vector.  Clarify this in the code.
    # Note: I also changed my scheme to move the atom.  So I should maybe expalin all that.
    adsorbate = config.adsorbate() # Note: Consider deleting this line.
    SystemSize=len(CopyConfig.anum) # Note: Consider deleting this line
    offset = adsorbate.centroid()
    adsorbate.coord = adsorbate.coord - offset
    adsorbate = RotateAxisAngle(adsorbate,RandAxis,RandRho)
    adsorbate.coord = adsorbate.coord + RandTrans + offset
    NewConfig = ConfigClass(CopyConfig) # Note: Make a notational choice of whether I deep copy or use the initializer.
    NewConfig.coord[SystemSize-config.AdsorbateSize:SystemSize] = adsorbate.coord
    return NewConfig

#====================================================================================
def moveR(config,Dummy,TransLimit,dummy):
# Returns a randomly pertubed configuration given an original configuration. Unlike
# the move() function above, this one moves every atom over a flat distribution in
# all direactions.
# Dear User: If you want to freeze some atoms or limit the movment of some
# atoms more tan others, just redefine the <Weight> accordingly.
# Note: consider adding option to move lattice vectors.
# Note: I'm getting sloppy with my capitalization convention.  Double think this.
# Note: The dummy is for compatibility between move() and moveR().  Think more about this later
#====================================================================================
    CopyConfig = copy.deepcopy(config)
    Weight = np.ones(len(CopyConfig.anum))
    ThetaTrans = np.random.rand(len(CopyConfig.anum)) * 2 * np.pi
    CosPhiTrans = np.random.rand(len(CopyConfig.anum)) * 2 - 1
    dR = np.zeros((CopyConfig.coord.shape))
    dR[:][0] = np.cos(ThetaTrans) * np.sqrt(1-CosPhiTrans**2) * Weight * TransLimit
    dR[:][1] = np.sin(ThetaTrans) * np.sqrt(1-CosPhiTrans**2) * Weight * TransLimit
    dR[:][2] = CosPhiTrans * Weight * TransLimit
    NewConfig = ConfigClass(CopyConfig) # Note: it feels weird that I am copying like this.  I should have a better way to do this.
    NewConfig.Coord = CopyConfig.Coord + dR
    return NewConfig

#====================================================================================
def shepherd(config):
# This function checks if the adsorbate centroid is more than 1 unit cell from the
# adsorbate centroid in X or Y, and, if so, moves it 1 unit cell back toward the centroid.
# Note: Impliment this into the actual cycle funcition.
# Note: This actually doesn't see mto be working that well so reconsider a way to make it actually on the midle microcell.
# Note: Fix this so it actually confines it to one unitcell.
#====================================================================================
    CopyConfig = ConfigClass(config)  # Note: I chose to do ConfigClass rather than deepcopy because it checks to make sure you are using ConfigClass and has error handling.
    try:
        from MC_Pref import SubLatticeGrid
    except ImportError:
        from MC_Defults import SubLatticeGrid
    # This defines a small lattice basis matrix and objects that represent the two bodies (adsorbant adsorbate):
    fence = config.LatticeMatrix / SubLatticeGrid
    adsorbent=ConfigClass(config.adsorbent())
    adsorbate=ConfigClass(config.adsorbate())
    SystemSize=len(CopyConfig.anum)
    # Then it computes the cetnroids and the vector difference between them:
    dCentroid = adsorbate.centroid() - adsorbent.centroid()
    # Then it projects the difference in centroids onto the lattice basis.
    coeff = np.linalg.solve(fence,dCentroid)
    # Then it gets rid of leading whole numbers on the lattice vector coefficients...
    # ...to move it back toward the center and then substacts the differnce in...
    # ...resulting vectors from the adsorbates position:
    ModCoeff = ((coeff+0.5)%1)-0.5
    yank = np.dot(fence,(ModCoeff-coeff))
    adsorbate.coord = adsorbate.coord + yank
    NewConfig = ConfigClass(CopyConfig) # Note: I might want to use deepcopy in these cases because it's faster and I don't have to check it.
    NewConfig.coord[SystemSize-config.AdsorbateSize:SystemSize] = adsorbate.coord # Note: Once again, it feels silly that I am copying twice like this.  Find a graceful way to get this done.
    return NewConfig

#====================================================================================
class Timer:
# This is an object that is used to count the cumulative time elapsted during certain
# operations like running SIESTA.  This is to keep track of performance.
# Note: I might want to make my own config class that you can add and initialize with another instance.
#====================================================================================
    class DoubleStart(Exception):
        pass

    class DoubleStop(Exception):
        pass

    def __init__(self):
        self.duration = 0.0
        self.tick = None
        self.timing = False

    def start(self):
        try:
            if self.timing:
                raise self.DoubleStart
            self.tick = time.time()
            self.timing = True
        except self.DoubleStart:
            sys.stderr.write( "You seem to have used start() twice without using stop().  No big deal, I will just pretend you didn't do that, but consider contacting the developer about this bug." )


    def update(self):
        if self.timing:
            self.duration = self.duration + time.time() - self.tick
            self.tick = time.time()

    def stop(self):
        try:
            if not self.timing:
                raise self.DoubleStop
            self.duration = self.duration + time.time() - self.tick
            self.timing = False
        except self.DoubleStop:
            sys.stderr.wite( "You seem to have used stop() twice without using start().  No big deal, I will just pretend you didn't do that, but consider contacting the developer about this bug." )

    # Adding two Timers will update both of them and return a timer with a duration
    # that is the sum of the duration of the two twrms, ticks at the time the addition
    # happens, and is timing if either of the two terms are timing.
    def __add__(self, other):
        self.update() # Note: Make sure I make deep copies if necessary.
        other.update()
        sum = Timer()   # Note: I'm not sure if I should call timer or self
        sum.tick = time.time()
        sum.timing = self.timing or other.timing
        sum.duration = self.duration + other.duration
        return sum

#====================================================================================
class Logger:
# This is an object that is used to log the performance of the MC program including
# the time spend on readding/writing files or running siesta and the number "hit rate"
# of the MC algorithm.  If the SubProcess and SuperProcess methods are both greater
# then zero, then summary() will also give you a value for parallel speedup as if
# the SuperProcess process is waiting for multiple process processes running in parallel.
# Note: by default use the subprocess I think
# Note: I need to find a way to prevent side effects on this.
#====================================================================================
    def __init__(self):
        self.siesta = Timer()
        self.ReadWrite = Timer()
        self.process = Timer()
        self.SuperProcess = Timer()
        self.hits = 0
        self.misses = 0

    def hit(self):
        self.hits = self.hits + 1

    def miss(self):
        self.misses = self.misses + 1

    def summary(self,FileName,FileLock = multiprocessing.Lock()):
        self.siesta.update()
        self.ReadWrite.update()
        self.process.update()
        self.SuperProcess.update()
        sum = 'Logger summary for Process ' + str(multiprocessing.current_process().pid) + ':\n'
        sum += 'Time spent on SIESTA: ' + str(self.siesta.duration) + 's'
        if self.process.duration > 0:
            sum += ' (' + str(self.siesta.duration/self.process.duration*100.0) + '%)'
        sum += '\nTime spent on reading and writing: ' + str(self.ReadWrite.duration) + 's'
        if self.process.duration > 0:
            sum += ' (' + str(self.ReadWrite.duration/self.process.duration*100.0) + '%)'
        sum += '\nTime of this process: ' + str(self.process.duration) + 's\n'
        if self.SuperProcess.duration > 0 :
            sum += 'Time of this super process: ' + str(self.SuperProcess.duration) + 's\n'
        if self.process.duration > 0 and self.SuperProcess.duration > 0:
            sum += 'Parallel speedup: ' + str(self.process.duration/self.SuperProcess.duration) + '\n'
        sum += 'Total hits: ' + str(self.hits)
        if (self.hits + self.misses) > 0:
            sum += ' (' + str(self.hits/float(self.hits+self.misses)*100.0) + '%)'
        sum += '\nTotal misses: ' + str(self.misses)
        if (self.hits + self.misses) > 0:
            sum += ' (' + str(self.misses/float(self.hits+self.misses)*100.0) + '%)'
        sum += '\n'
        FileLock.acquire()
        with open(FileName,'a') as WriteFile: # Note: make sure this isn't supposed to be ea capital W
            WriteFile.write(sum)
            WriteFile.close()
        FileLock.release()

    def __add__(self,other):
        sum = Logger()
        sum.siesta = self.siesta + other.siesta
        sum.ReadWrite = self.ReadWrite + other.ReadWrite
        sum.process = self.process + other.process
        sum.hits = self.hits + other.hits
        sum.misses = self.misses + other.misses
        sum.SuperProcess = self.SuperProcess + other.SuperProcess
        return sum
        # Note: Add some error handeling here.
