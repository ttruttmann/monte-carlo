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
# If the user does not have statsmodels, then they must manually attach the adolecule
# to the adsorbent in the file InputGeom.xyz:
try: import statsmodels.api as sm
except ImportError: pass
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
def bias(config, history, MetaHeight, MetaWidths):
# Funciton that generates a bias based on user prefrences and sampling history. 
#====================================================================================
# Note: Some day I might make the Gaussians spill over to the next unit cell, or the next rotational 
# element on a sphere.  I might also make the guassians have a equal span in linear distance 
# on the surface of a sphere. 
    try:
        present = config.FloppyCoord()
        BiasEnergy = np.sum((MetaHeight * np.exp(-1/2*(present-history)**2/MetaWidths**2)))
        return BiasEnergy
    except:
        raise

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
# Raised when there is a linear angle that is interfering with angle calculations: 
class LinearAngleError(Exception): pass
# Raised when SIESTA returns an error:
class SiestaError(Exception): pass
    
#====================================================================================
def angle(v1,v2,PosN = None):
# This funciton returns the angle between two 3-vectors.  It returns degrees. 
# If PosN is specified, If the cross between the two vectors is more-or-less in the 
# opposite direction as <PosN>, then the returned angle is negative.  
#====================================================================================  
    try: 
        if (v1.size != 3) or (v2.size != 3):
            raise ValueError
        TheCos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        # Occasionally, TheSin will be slighter greater than 1 due to machine error.
        # I added a quick condition to fix this:
        if TheCos > 1:
            radians = 0.
        else:
            radians = np.arccos(TheCos)
        degrees = np.rad2deg(radians)
        # However if a normal vector <PosN> is supplied, the angle maybe be negative
        # according to the right hand rule.
        if not np.array_equal(PosN, None):
            if np.dot( np.cross(v1,v2) ,PosN) < 0:
                degrees = -degrees
            elif (np.dot( np.cross(v1,v2), PosN) == 0) and (np.linalg.norm(np.cross(v1,v2)) > 0):
                raise LinearAngleError
        return degrees
    except ValueError: 
        sys.stderr.write("The vectors must be 3 dimensions for angle to work.")
        raise
    except LinearAngleError:
        sys.stderr.write("You chose a refrence normal vector that is perpendicular to the actual cross vector.  You surely made a mistake." )
        raise

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
#   AdmoleculeSize:      An integer storing the number of atoms in the admolecule.
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
#                       adsorbent. Only works if AdmoleculeSize is properly defined.
#
#   admolecule():        Returns another ConfigClass that only contains the atoms of the
#                       ad-molecule. Only works if AdmoleculeSize is properly defined.
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
#   FloppyCoord():      Returns the centroid of the admolecule as well as the euler
#                       angles defining the orientation of the admolecule with repsect
#                       to the global xyz coordinate system.  See the code for
#                       how the angles are determined.  
#
#====================================================================================
    def __init__(self,father=None):
        # If there is no argument, an empty default ConfigClass is created:
        if father is None:
            self.anum = np.zeros(0 , dtype = int)
            self.coord = np.zeros((0,3), dtype=float)
            self.E = 0.0
            self.AdmoleculeSize = None
            self.LatticeMatrix = None
            self.SubLatticeGrid = None
        # If the argument is a string, then it attempts to read it as an xyz file:
        elif type(father) is str:
            try:
                # First some defaults are set that are not available from the xyz file:
                self.E = 0.0
                self.AdmoleculeSize = None
                self.LatticeMatrix = None
                self.SubLatticeGrid = None
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
                self.AdmoleculeSize = CopyConfig.AdmoleculeSize
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
    # AdmoleculeSize is properly defined:
    def adsorbent(self):
        # First it makes a copy of self and declares a new ConfigClass:
        TotalSystem = copy.deepcopy(self)
        NewConfig = ConfigClass()
        # Then is assigns values to attributes appropriately:
        NewConfig.anum = TotalSystem.anum[: len(TotalSystem.anum)-TotalSystem.AdmoleculeSize]
        NewConfig.coord = TotalSystem.coord[: len(TotalSystem.coord)-TotalSystem.AdmoleculeSize]
        NewConfig.E = 0.0
        NewConfig.AdmoleculeSize = TotalSystem.AdmoleculeSize
        NewConfig.LatticeMatrix = None
        NewConfig.SubLatticeGrid = None
        return NewConfig
        
    # Returns a new ConfigClass instance that represents the admolecule. Only works 
    # if AdmoleculeSize is properly defined:
    def admolecule(self):
        TotalSystem = copy.deepcopy(self)
        NewConfig = ConfigClass()
        NewConfig.anum = TotalSystem.anum[len(TotalSystem.anum)-TotalSystem.AdmoleculeSize :]
        NewConfig.coord = TotalSystem.coord[len(TotalSystem.coord)-TotalSystem.AdmoleculeSize:]
        NewConfig.E = 0.0
        NewConfig.AdmoleculeSize = TotalSystem.AdmoleculeSize
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
        buffer += 'energy = np.array((),dtype=float)\n'
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
        buffer += 'TempConfig.anum = ' + repr(self.anum) + '\n'
        buffer += 'TempConfig.coord = ' + repr(self.coord) + '\n'
        buffer += 'TempConfig.E = ' + repr(self.E) + '\n'
        buffer += 'TempConfig.AdmoleculeSize = ' + repr(self.AdmoleculeSize) + '\n'
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
        
    # Generates a new file to record the floppy coordinates to:
    def StartWriteFloppy(self, FileName, lock = multiprocessing.Lock()):
        # First it checks for size inconsistencies:
        self.verify()
        # Then it imports numpy:
        buffer = 'import numpy as np\n'
        # Then it initializes an empty numpy array:
        buffer += 'FloppyCoord = np.zeros((0,6))\n'
        # Then it writes the buffer to the file:
        lock.acquire()
        file = open(FileName,'w')
        file.write(buffer)
        file.close()
        lock.release()
        return
        
    # Appends Floppy coordinate data to a python-readable file:    
    def WriteFloppy(self,FileName,lock = multiprocessing.Lock()):
        # First it checks for size inconsistencies:
        self.verify()
        # Then it directly writes the energy to the file:
        lock.acquire()
        file = open(FileName,'a')
        file.write('FloppyCoord = np.append(FloppyCoord,' + 'np.' + repr(self.FloppyCoord()) + ')\n')
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
            self.StartWriteFloppy(CommandHome + '/output/FloppyEnsemble.py',lock = lock)
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
            self.WriteFloppy(CommandHome + '/output/FloppyEnsemble.py', lock = lock)
            return
        except AttributeError:
             sys.stdout.write('The first argument of ConfigClass().StartWrite() must be a string\n')
             raise
             
    def FloppyCoord(self):
        # For the sake of clarity, my comments refer to global coredinates as 
        # X,Y, and Z, and the local coordinates as x', y', and z'. 
        try:
            # First I define an empty six vector that will be edited and returned at end:
            FloppyVec = np.zeros(6)
            # Then the first three elements are filled with the centroid. 
            FloppyVec[0:3] = self.centroid()
            # Then the two vectors are created from the admolecule: 
            admol = self.admolecule()
            vec1 = admol.coord[1] - admol.coord[0]
            vec2 = admol.coord[2] - admol.coord[1]
            # Then it checks if the angle between them is near linear:
            if angle(vec1,vec2) > 179 or angle(vec1,vec2) < 1: 
                raise LinearAngleError
            # Then vec1 is normalized and becomes x' 
            LocalX = vec1 / np.linalg.norm(vec1)
            # Then the component of vec2 that is perpendicular to x' is normalized 
            # and used as y': 
            LocalY = vec2 - np.dot(vec2,LocalX) * LocalX
            LocalY = LocalY / np.linalg.norm(LocalY)
            # Then I get z' simply by crossing x' and y'
            LocalZ = np.cross(LocalX,LocalY)
            # Then I define vector N as a means to find the proper Euler angles:
            # In most cases, vecN is simply a cross product of Z and z':
            if not np.array_equal(LocalZ, np.array((0,0,1))):     
                vecN = np.cross(LocalZ, np.array((0,0,1)) )
            # However if z' points in the Z direction, then vecN will be indentically 
            # zero. This is not acceptable.  To combat this, we default vecN to the x' direction:
            else:
                vecN = LocalX
            # alpha is the angle between N and X.  I assign this to the 4th element of FloppyVec:
            FloppyVec[3] = angle(np.array((1,0,0)), vecN, PosN = np.array((0,0,1)))
            # beta is the angle between Z and z'.  I assign this to the 5th element of FloppyVec:
            FloppyVec[4] = angle(np.array((0,0,1)), LocalZ)
            # gamma is the angle between N and x'. I assign this the the 6th element of FloppyVec:
            FloppyVec[5] = angle(vecN, LocalX, PosN = LocalZ)
            # Then I return that 6-vector FloppyVec.  This contains all the coordinates
            # that are floppy in a simple admolecule system.
            # Then when I return the vector, I pad it in brackets so that it is a 2=rank
            # tansor to make the concatenate() happy on the other end.
            return np.array([FloppyVec])
        except LinearAngleError:
            print ("The angle between the first 3 atoms in your admolecule is very close to linear.  Consider changing the order of your admolecule atoms in your SIESTA input file (sorry).") 
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
def MC_Cycle(StartConfig, EnsembleSize, CommandHome, ScratchPath, WriteLock, LogQ, baseline):
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
    try:
        from MC_Pref import MetaDynamicsOn
    except:
        from MC_Defaults import MetaDynamicsOn
    try:
        from MC_Pref import MetaWidths
    except:
        from MC_Defaults import MetaWidths
    try:
        from MC_Pref import MetaHeight
    except:
        from MC_Defaults import MetaHeight
    try:
        from MC_Pref import CounterpoiseOn
    except:
        from MC_Defaults import CounterpoiseOn

    CycleLogger = Logger()
    CycleLogger.process.start()
    # First the variables defined globally are declared in the funciton and scientific
    # constants are defined:

    k = 8.6173324e-5 # in eV K^(-1) from Wikipedia.
    # I also define an empty 2-rank tensor that will store the past positions of the admolecule: 
    history = np.zeros((0,6))
    # Note: In the future I am interested in trashing multiprocessing (it doesn't work for
    # nodes anyways) and getting Parallel Python. I am also thinking about getting ride of 
    # EnsembleSize and just using the size of history. (But history has to be shared first)
    # Then it begins the process of defining new structures and testing them:
    OldConfig = ConfigClass(StartConfig)
    OldBiasE = 0.
    T_index = [0,0]
    while True:
        # First it handles the simulated anealing sequence:\
        T = T_Sequence[T_index[0]][1]
        T_index[1] += 1
        if T_index[1] == T_Sequence[T_index[0]][0]:
            T_index[1] = 0
            T_index[0] += 1
            if T_index[0] == len(T_Sequence):
                T_index[0] = 0
        CandConfig = move(OldConfig,TransLimit,RhoLimitDeg)
        CandConfig = shepherd(CandConfig) if ShepherdOn else CandConfig
        CandConfig.E = AdEnergy(CandConfig,CommandHome,NumberOfSiestaCores,CycleLogger,CounterpoiseOn,baseline,lock = multiprocessing.Lock())
        if MetaDynamicsOn:
            CandBiasE = bias(CandConfig, history, MetaHeight, MetaWidths)
            CandBiasFactor = np.exp(-(CandBiasE-OldBiasE)/(k*T))
        else:
            CandBiasFactor = 1
        rand = np.random.random()
        prob = CandBiasFactor * np.exp(-(CandConfig.E-OldConfig.E)/(k*T))
        if rand <= prob:
            OldConfig = ConfigClass(CandConfig) # This is a way to assign without side-effects
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
            if Optimize:
                TransLimit = TransLimit * 0.95
                RhoLimitDeg = RhoLimitDeg * 0.95
        CycleLogger.ReadWrite.start()
        OldConfig.Write(CommandHome, lock = WriteLock)
        CycleLogger.ReadWrite.stop()
        EnsembleSize.value = EnsembleSize.value + 1
        # Recording the position for future metadynamics biasing:            
        history = np.concatenate((history,OldConfig.FloppyCoord()),axis=0)
        # Then I redefine the bias for the OldConfig using the OldConfig's new (or old)
        # geometry and the new history. 
        OldBiasE = bias(OldConfig, history, MetaHeight, MetaWidths)
        CycleLogger.ReadWrite.start()
        CycleLogger.summary(CommandHome + '/time.log',WriteLock)
        CycleLogger.ReadWrite.stop()

#====================================================================================
def SIESTA(config,CommandHome,NumberOfSiestaCores,Logger,ghost=None):
# This function uses the file template.fdf in the SiestaFiles directory as a template
# to run a SIESTA job to compute the potential energy.  It appends the geometric
# coordinates to the end of the file in the form "X\tY\tZ\tN" where N is the species
# number (which is extracted from template.fdf.  If you want to change the energy
# calculation parameters, edit the template.fdf file.
#====================================================================================
    # First I ensure that <ghost> is defined properly:
    try:
        if (ghost != None) and (ghost != 'adsorbent') and (ghost != 'admolecule'):
            raise ValueError
    except ValueError:
        sys.stdout.write("ghost must be set to None, 'adsorbent', or 'admolecule'.")
    # Then the working directory <WorkDir> is created, and the pseudopotentials
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
    Logger.ReadWrite.start()
    with open(WorkDir + '/template.fdf','r') as file:
        SpeciesNumbers = dict()
        NoteTaking = False
        for line in file:
            if '%block ChemicalSpeciesLabel' in line:
                NoteTaking = True
                continue
            if '%endblock ChemicalSpeciesLabel' in line:
                break
            if NoteTaking == True:
                Note = line.replace('\n','').replace('\t','').replace(',','').split(' ')
                while '' in Note:
                    Note.remove('')
                SpeciesNumbers[Note[1]] = Note[0]
        file.close()
    Logger.ReadWrite.stop()
    # Then we correct the NumberOfAtoms field in the .fdf file: 
    # Explain in user manual that this line must be blank
    os.system("sed -i -e 's/NumberOfAtoms/NumberOfAtoms          " + str(len(config.anum)) + "/g' " + WorkDir + "/template.fdf")
    # Then we define an output buffer <OutBuffer> in memory to store the string that
    # that we will write to the file:
    OutBuffer = '%block AtomicCoordinatesAndAtomicSpecies\n'
    # We split the buffer into the the adsorbent and the admolecule (two loops):
    # Loop for adsorbent:
    for i in range(len(config.adsorbent().anum)):
        for j in range(3):
            OutBuffer += str(config.adsorbent().coord[i][j]) + '\t'
        # If we are ghosting these the adsorbent, we must just write a zero
        if ghost == 'adsorbent':
            OutBuffer += SpeciesNumbers[str(-config.adsorbent().anum[i])]
        else:
            OutBuffer += SpeciesNumbers[str(config.adsorbent().anum[i])]
        OutBuffer += '\n'
    # Loop for admolecule:
    for i in range(len(config.admolecule().anum)):
        for j in range(3):
            OutBuffer += str(config.admolecule().coord[i][j]) + '\t'
        # If we are ghosting the admolecule, we include the ghost atom:
        if ghost == 'admolecule':
            OutBuffer += SpeciesNumbers[str(-config.admolecule().anum[i])]
        else:
            OutBuffer += SpeciesNumbers[str(config.admolecule().anum[i])]
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
    status = os.system('mpirun -np ' + str(NumberOfSiestaCores) + ' siesta < template.fdf | tee template.out')
    Logger.siesta.stop()
    os.chdir('..')
    # Then we need to raise an error if Siesta has an error: 
    if status != 0:
        sys.stderr.write('SIESTA returned an error.')
        raise SiestaError
    #Then we extract the energy from the output file:
    Energy = 0.00
    # Dear User: This only works for harris functional.  If you edit <template.fdf>,
    # please replace 'siesta: Eharris(eV) =' appropriately.
    Logger.ReadWrite.start()
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
    return Energy


#====================================================================================
def AdEnergy(config,CommandHome,NumberOfSiestaCores,Logger,CounterpoiseOn,baseline,lock = multiprocessing.Lock()):
# This function takes the configuration and the 'baseline' energy of the species seperatre
# and returns the adsorbtion energy.  If <CounterpoiseOn = True>, then a Counterpoise 
# calcualtion is done automatically and the basis set superposition energy is recorded
# in the a file named "output/SuperpositionError.csv". 
#====================================================================================
# Note: Consider jsut getting rid all the passed arguments and picking up the 
# global variable names.  But you would probably have to do this for the SIESTA function too.
    # If counterpoise is turned off, then we simply subtract the system energy frome the baseline:
    if CounterpoiseOn == False:
        return SIESTA(config,CommandHome,NumberOfSiestaCores,Logger,ghost=None) - baseline
    # If counterpoise is turned on, then the calculation is a little more complicated:
    if CounterpoiseOn:
        GhostAdsorbent  = SIESTA(config,CommandHome,NumberOfSiestaCores,Logger,ghost='adsorbent')
        GhostAdmolecule = SIESTA(config,CommandHome,NumberOfSiestaCores,Logger, ghost='admolecule')
        NoGhost         = SIESTA(config,CommandHome,NumberOfSiestaCores,Logger, ghost=None)
        AdEnergy = NoGhost - (GhostAdsorbent + GhostAdmolecule)
        # We also keep track of the basis-set-superposition error for recording purposes:
        SuperError = baseline - (GhostAdsorbent + GhostAdmolecule)
        buffer = '\n' + str(SuperError)
        Logger.ReadWrite.start()
        lock.acquire()
        file = open(CommandHome + '/output/SuperError.csv','a')
        file.write(buffer)
        file.close()
        lock.release()
        Logger.ReadWrite.stop()
        return AdEnergy

#====================================================================================
def attach(adsorbent,admolecule,RightSide=None):
# Attached the admolecule to the adsorbent. Attaches to side that <norm> points in if
# supplied.  This will only work for surfaces that are reletively thin and flat. 
# <RightSide> should be 3-element numpy array
#====================================================================================
    # First make coppies of the arguments to prevent side effects: 
    admolecule = copy.deepcopy(admolecule)
    adsorbent = copy.deepcopy(adsorbent)    
    # Then perform 3 linear regressions on the adsorbent, and take the one with the 
    # lowest error. This represents it as a plane: 
    x = adsorbent.coord[:,0]
    y = adsorbent.coord[:,1]
    z = adsorbent.coord[:,2]
    Norms = np.zeros((3,3))
    RMSError = np.zeros(3)
    #Regression 1:
    X = np.column_stack((y,z))
    X = sm.add_constant(X) # Design matrix
    Z = x # Dependent variable
    results = sm.OLS(Z,X).fit()
    b = results.params[0]
    m1 = results.params[1]
    m2 = results.params[2]
    Norms[0] = np.array((1/b,-m1/b,-m2/b))
    RMSError[0] = np.linalg.norm(results.bse)
    #Regression 2:
    X = np.column_stack((z,x))
    X = sm.add_constant(X) # Design matrix
    Z = y # Dependent variable
    results = sm.OLS(Z,X).fit()
    b = results.params[0]
    m1 = results.params[1]
    m2 = results.params[2]
    Norms[1] = np.array((-m2/b,1/b,-m1/b))
    RMSError[1] = np.linalg.norm(results.bse)
    #Regression 1:
    X = np.column_stack((x,y))
    X = sm.add_constant(X) # Design matrix
    Z = z # Dependent variable
    results = sm.OLS(Z,X).fit()
    b = results.params[0]
    m1 = results.params[1]
    m2 = results.params[2]
    Norms[2] = np.array((-m1/b,-m2/b,1/b))
    RMSError[2] = np.linalg.norm(results.bse)
    # Then we just choose the norm with the lease RMS error and we normalize:
    Norm = Norms[np.argmin(RMSError)]
    Norm = Norm / np.linalg.norm(Norm)
    # If the user specified a side, we will orient the Norm in that direction.
    if RightSide != None:
        Norm = np.dot(Norm,RightSide) / np.linalg.norm(RightSide)
    # Then we project all adsorbent atoms' positions of the adsorbent onto the norm:
    AdsorbentDist = np.dot(adsorbent.coord,Norm)
    AdsorbentDist = AdsorbentDist - np.average(AdsorbentDist)
    # We do the same with the admolecule:
    AdmoleculeDist = np.dot(admolecule.coord,Norm)
    AdmoleculeDist= AdmoleculeDist - np.average(AdmoleculeDist)
    # Then we use the max of the AdsorbentDist and the min of the AdmoleculeDist
    # and add 1.77 to as the distance between the two centroids at the beginning of the 
    # Lennard-Jones optimization (1.77 comes from Zimmerman et all. [1])
    InterDist = np.max(AdsorbentDist) - np.min(AdmoleculeDist) + 1.77
    # Then we have to define an energy function based on a 6-12 potential: 
    def QuickEnergy(InterDist):
        R_eq = 1.77
        TransVec = (adsorbent.centroid() + Norm * InterDist) - admolecule.centroid()
        NewAdMoleculeCoords = admolecule.coord + TransVec
        InterDistances = np.zeros(0)
        # Note: Vectorize this in the future if possible.
        for iAtom in adsorbent.coord:
            for jAtom in NewAdMoleculeCoords:
                InterDistances = np.append(InterDistances,np.linalg.norm(iAtom - jAtom))
        Energies = (R_eq/InterDistances)**12 - 2*(R_eq/InterDistances)**6
        return np.sum(Energies)
    # Then I minimize the energy function to find best distance using binarry search:
    # We set up "inner" and "outer" walls
    # Note: This may not work with strange adrorbent geometries:
    inner = 0.0
    outer = InterDist
    OuterEnergy = QuickEnergy(outer)
    # Then I move the walls in closer to the minimum in 15 steps:
    for i in range(15):
        # We see what the energy half way between the walls is:
        MiddleEnergy = QuickEnergy(np.average((inner,outer)))
        # If the energy is higher, we move the inner wall to the middle:
        if MiddleEnergy >= OuterEnergy:
            inner = np.average((inner,outer))
            continue
        elif MiddleEnergy < OuterEnergy:
        # If the energy is lower, we move the outer wall there: 
            outer = np.average((inner,outer))
            OuterEnergy = MiddleEnergy
            continue
    # Then we use the final outer wall as the distance:
    InterDist = outer
    # Then I change the position of the admolecule:
    TransVec = (adsorbent.centroid() + Norm * InterDist) - admolecule.centroid()
    admolecule.coord = admolecule.coord + TransVec
    # Then I fuse the two together and return the new object. 
    adsorbent.coord = np.append(adsorbent.coord,admolecule.coord,axis=0)
    adsorbent.anum = np.append(adsorbent.anum,admolecule.anum)
    adsorbent.AdmoleculeSize = len(admolecule.anum)
    return ConfigClass(adsorbent)

#====================================================================================
def move(config,TransLimit,RhoLimitDeg):
# Returns a randomly pertubed configuration given an original configuration. This
# function works with any single molecule absorbate system as long as the admolecule
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
    admolecule = config.admolecule() # Note: Consider deleting this line.
    SystemSize=len(CopyConfig.anum) # Note: Consider deleting this line
    offset = admolecule.centroid()
    admolecule.coord = admolecule.coord - offset
    admolecule = RotateAxisAngle(admolecule,RandAxis,RandRho)
    admolecule.coord = admolecule.coord + RandTrans + offset
    NewConfig = ConfigClass(CopyConfig) # Note: Make a notational choice of whether I deep copy or use the initializer.
    NewConfig.coord[SystemSize-config.AdmoleculeSize:SystemSize] = admolecule.coord
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
# This function checks if the admolecule centroid is more than 1 unit cell from the
# admolecule centroid in X or Y, and, if so, moves it 1 unit cell back toward the centroid.
# Note: Impliment this into the actual cycle funcition.
# Note: This actually doesn't see mto be working that well so reconsider a way to make it actually on the midle microcell.
# Note: Fix this so it actually confines it to one unitcell.
#====================================================================================
    CopyConfig = ConfigClass(config)  # Note: I chose to do ConfigClass rather than deepcopy because it checks to make sure you are using ConfigClass and has error handling.
    try:
        from MC_Pref import SubLatticeGrid
    except ImportError:
        from MC_Defults import SubLatticeGrid
    # This defines a small lattice basis matrix and objects that represent the two bodies (adsorbant admolecule):
    fence = config.LatticeMatrix / SubLatticeGrid
    adsorbent=ConfigClass(config.adsorbent())
    admolecule=ConfigClass(config.admolecule())
    SystemSize=len(CopyConfig.anum)
    # Then it computes the cetnroids and the vector difference between them:
    dCentroid = admolecule.centroid() - adsorbent.centroid()
    # Then it projects the difference in centroids onto the lattice basis.
    coeff = np.linalg.solve(fence,dCentroid)
    # Then it gets rid of leading whole numbers on the lattice vector coefficients...
    # ...to move it back toward the center and then substacts the differnce in...
    # ...resulting vectors from the adsorbates position:
    ModCoeff = ((coeff+0.5)%1)-0.5
    yank = np.dot(fence,(ModCoeff-coeff))
    admolecule.coord = admolecule.coord + yank
    NewConfig = ConfigClass(CopyConfig) # Note: I might want to use deepcopy in these cases because it's faster and I don't have to check it.
    NewConfig.coord[SystemSize-config.AdmoleculeSize:SystemSize] = admolecule.coord # Note: Once again, it feels silly that I am copying twice like this.  Find a graceful way to get this done.
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

# References: 
# [1] Zimmerman, Paul M., Martin Head-Gordon, and Alexis T. Bell. "Selection and validation 
# of charge and Lennard-Jones parameters for QM/MM simulations of hydrocarbon interactions # with zeolites." Journal of chemical theory and computation 7.6 (2011): 1695-1703.