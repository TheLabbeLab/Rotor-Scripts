# program begins

import sys, math, os, shutil
from collections import Counter
from io import open
import numpy as np
import zmat2xyz

## -- CONSTANTS -- ##

# threshold beyond average of covalent radiii to determine bond cutoff
bond_thresh = 1.2

# conversion from radians to degrees and vice versa
rad2deg = 180.0 / math.pi
deg2rad = 1.0 / rad2deg

# covalent (or ionic) radii by atomic element (Angstroms) from
# "Inorganic Chemistry" 3rd ed, Housecroft, Appendix 6, pgs 1013-1014
cov_rads = {  'H' : 0.37, 'C' : 0.77, 'O' : 0.73, 'N' : 0.75, 'F' : 0.71,
  'P' : 1.10, 'S' : 1.03, 'Cl': 0.99, 'Br': 1.14, 'I' : 1.33, 'He': 0.30,
  'Ne': 0.84, 'Ar': 1.00, 'Li': 1.02, 'Be': 0.27, 'B' : 0.88, 'Na': 1.02,
  'Mg': 0.72, 'Al': 1.30, 'Si': 1.18, 'K' : 1.38, 'Ca': 1.00, 'Sc': 0.75,
  'Ti': 0.86, 'V' : 0.79, 'Cr': 0.73, 'Mn': 0.67, 'Fe': 0.61, 'Co': 0.64,
  'Ni': 0.55, 'Cu': 0.46, 'Zn': 0.60, 'Ga': 1.22, 'Ge': 1.22, 'As': 1.22,
  'Se': 1.17, 'Kr': 1.03, 'X' : 0.00}

## -- END OF CONSTANTS -- ##

## -- IO FUNCTIONS -- ##

# read file data into a 2-d array
def get_file_string_array(file_name):
    #Opens file for array
    try:
        file = open(file_name, "r")
    except IOError:
        print('Error: file (%s) not found!\n' % (file_name))
        sys.exit()
    lines = file.readlines()
    #Opens file for string
    try:
        file_1 = open(file_name, "r")
    except IOError:
        print('Error: file (%s) not found!\n' % (file_name))
        sys.exit()
    file_content = file_1.read()
    file.close()
    file_1.close()
    array = []
    for line in lines:
        array.append(line.split())
    return array, file_content

# read in geometry from xyz file
def get_geom(xyz_array):
    # xyz_array = get_file_string_array(xyz_file_name)
    n_atoms = int(xyz_array[0][0])
    at_types = ['' for i in range(n_atoms)]
    coords = [[0.0 for j in range(3)] for i in range(n_atoms)]
    for i in range(n_atoms):
        at_types[i] = xyz_array[i+2][0]
        for j in range(3):
            coords[i][j] = float(xyz_array[i+2][j+1])
    geom = [at_types, coords]
    return geom

# input syntax and usage warnings
def get_inputs():
    if (not len(sys.argv) == 2):
        print('Usage: torsions.py XYZ_FILE\n')
        print('  XYZ_FILE: coordinates of target molecule\n')
        sys.exit()
    else:
        xyz_file_name = sys.argv[1]
        return xyz_file_name

# print list of torsion angles to screen
def print_torsions(geom, torsions):
    at_types = geom[0]
    n_torsions = len(torsions)
    print('%i torsion(s) found (degrees)' % (n_torsions))
    for q in range(n_torsions):
        n1, n2, n3, n4 = torsions[q][0:4]
        t1234 = torsions[q][4]
        nstr = '%i-%i-%i-%i' % (n1+1, n2+1, n3+1, n4+1)
        tstr = '(%s-%s-%s-%s) ' % (at_types[n1], at_types[n2], at_types[n3], at_types[n4])
        print(' %-15s  %-13s  %8.3f\n' % (nstr, tstr, t1234), end='')
    print('\n', end='')

## -- END OF IO FUNCTIONS -- ##


## -- MATHS FUNCTIONS -- ##

# calculate distance between two 3-d cartesian coordinates
def get_r12(coords1, coords2):
    r2 = 0.0
    for p in range(3):
        r2 += (coords2[p] - coords1[p])**2
    r = math.sqrt(r2)
    return r

# calculate unit vector between to 3-d cartesian coordinates
def get_u12(coords1, coords2):
    r12 = get_r12(coords1, coords2)
    u12 = [0.0 for p in range(3)]
    for p in range(3):
        u12[p] = (coords2[p] - coords1[p]) / r12
    return u12

# calculate dot product between two unit vectors
def get_udp(uvec1, uvec2):
    uvec1 = np.array(uvec1)
    uvec2 = np.array(uvec2)
    udp = np.dot(uvec1, uvec2)
    return udp

# calculate unit cross product between two unit vectors
def get_ucp(uvec1, uvec2):
    ucp = np.cross(uvec1, uvec2)
    return ucp


# calculate angle between three 3-d cartesian coordinates
def get_a123(coords1, coords2, coords3):
    u21 = get_u12(coords2, coords1)
    u23 = get_u12(coords2, coords3)
    dp2123 = get_udp(u21, u23)
    a123 = rad2deg * math.acos(dp2123)
    return a123

# calculate torsion angle between four 3-d cartesian coordinates
def get_t1234(coords1, coords2, coords3, coords4):
    u21 = get_u12(coords2, coords1)
    u23 = get_u12(coords2, coords3)
    u32 = get_u12(coords3, coords2)
    u34 = get_u12(coords3, coords4)
    u21c23 = get_ucp(u21, u23)
    u32c34 = get_ucp(u32, u34)
    dp = get_udp(u21c23, u32c34)
    sign = 2 * float(get_udp(u21c23, u34) < 0) - 1
    t1234 = rad2deg * sign * math.acos(dp)
    return t1234

## -- END OF MATHS FUNCTIONS -- ##


## -- TOPOLOGY FUNCTIONS -- ##

# build graph of which atoms are covalently bonded
def get_bond_graph(geom, bond_thresh):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    double_bonds = []
    bond_graph = [[] for i in range(n_atoms)]
    for i in range(n_atoms):
        covrad1 = cov_rads[at_types[i]]
        for j in range(i+1, n_atoms):
            covrad2 = cov_rads[at_types[j]]
            thresh = bond_thresh * (covrad1 + covrad2)
            # adjusts threshhold to catch double bonds
            if bond_thresh == 1.2:
                double_bond_thresh = 1.2 * (covrad1*0.73 + covrad2*0.73)
            if bond_thresh == 1.4:
                double_bond_thresh = 1.2 * (covrad1*0.74 + covrad2*0.74)
            r12 = get_r12(coords[i], coords[j])
            if (r12 < thresh):
                bond_graph[i].append(j)
                bond_graph[j].append(i)
            if (r12 < double_bond_thresh):
                double_bonds.append([i, j])
    return bond_graph, double_bonds

# determine atoms which are covalently bonded from bond graph
def get_bonds(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    bonds = []
    for i in range(n_atoms):
        for a in range(len(bond_graph[i])):
            j = bond_graph[i][a]
            if (i < j):
                r12 = get_r12(coords[i], coords[j])
                bonds.append([i, j, r12])
    return bonds

# determine atoms which form a bond angle from bond graph
def get_angles(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    angles = []
    for j in range(n_atoms):
        n_jbonds = len(bond_graph[j])
        for a in range(n_jbonds):
            i = bond_graph[j][a]
            for b in range(a+1, n_jbonds):
                k = bond_graph[j][b]
                a123 = get_a123(coords[i], coords[j], coords[k])
                angles.append([i, j, k, a123])
    return angles

# determine atoms which form torsion angles from bond graph
def get_torsions(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    torsions = []
    for j in range(n_atoms):
        n_jbonds = len(bond_graph[j])
        for a in range(n_jbonds):
            k = bond_graph[j][a]
            if (k < j):
                continue
            n_kbonds = len(bond_graph[k])
            for b in range(n_jbonds):
                i = bond_graph[j][b]
                if (i == k):
                    continue
                for c in range(n_kbonds):
                    l = bond_graph[k][c]
                    if (l == j or l == i):
                        continue
                    t1234 = get_t1234(coords[i], coords[j], coords[k], coords[l])
                    torsions.append([i, j, k, l, t1234])
    return torsions

# find axis where the second atom is not bonded to anything else! Extract that axis and redo!
# only caveat is that heavy atoms must bethe first ones in zmatrix!
def include_cyclics_branched(axes):
    rotor_axes = []
    check = True
    counter = 0
    while (check == True):
        for axis in axes:
            second_atom = axis[-1]
            first_atom = axis[0]
            counter_second = 0
            counter_first = 0
            for ax in axes:
                if second_atom in ax:
                    counter_second += 1
                if first_atom in ax:
                    counter_first += 1
            if counter_second == 1 or counter_first == 1:
                rotor_axes.append(axis)

        new_axes = [axis for axis in axes if axis not in rotor_axes]
        if axes == new_axes:
            check = False
        axes = new_axes
    if len(axes)>1:
        TS_ring = True
    else:
        TS_ring = False

    return rotor_axes, TS_ring

def get_all_rotors(bond_graph, bond_graph_woTSthresh=[]):
    #Fixes numbering system to match Gaussian
    from copy import deepcopy

    gauss_bond_graph_woTSthresh = deepcopy(bond_graph_woTSthresh)
    for bonds in gauss_bond_graph_woTSthresh:
        for i in range(len(bonds)):
            bonds[i] += 1

    gauss_bond_graph = deepcopy(bond_graph)
    for bonds in gauss_bond_graph:
        for i in range(len(bonds)):
            bonds[i] += 1

    #Heavy atoms, more than 1 bond
    heavy_atoms = [[atom, idx+1] for idx, atom in enumerate(gauss_bond_graph) if len(atom)>1]

    #'Heavy-heavy atoms' is bond graph of heavy atom bonds alone!
    heavy_heavy_atoms = []
    for heavy_atom in heavy_atoms:
        heavy_heavy_atom = [atom for atom in heavy_atom[0] if len(gauss_bond_graph[atom-1])>1]
        heavy_heavy_atoms.append([heavy_heavy_atom, heavy_atom[1]])

    #General axes from all heavy-heavy bonds
    general_axes = []
    for i in range(len(heavy_heavy_atoms)):
        for j in range(len(heavy_heavy_atoms[i][0])):
            general_axes.append([heavy_heavy_atoms[i][-1], heavy_heavy_atoms[i][0][j]])

    #Unique list of all heavy atom bonds - rotor axes for branched molecules included here!!!
    axes = []
    for axis in general_axes:
        for element in general_axes:
            if sorted(axis) == sorted(element):
                if sorted(axis) not in axes:
                    axes.append(axis)

    rotor_axes, TS_ring = include_cyclics_branched(axes)
    # alternative approach, not used at the moment (might be useful later on)
    # if not TS_ring and len(bond_graph_woTSthresh)>0:
    #     #
    #     heavy_atoms = [[atom, idx+1] for idx, atom in enumerate(gauss_bond_graph) if len(atom)>1]
    #     heavy_atomswo = [[atom, idx+1] for idx, atom in enumerate(gauss_bond_graph_woTSthresh) if len(atom)>1]
    #
    #     heavy_heavy_atoms = []
    #     for heavy_atom in heavy_atoms:
    #         heavy_heavy_atom = [atom for atom in heavy_atom[0] if len(gauss_bond_graph_woTSthresh[atom-1])>1]
    #         heavy_heavy_atoms.append([heavy_heavy_atom, heavy_atom[1]])
    #
    #     ## If there is no TS ring, rotor axes are found using a different approach (not based on bond elimination,
    #     ## instead compared bond graphs at diff thresholds)
    #     # alternative approach
    #     general_axes = []
    #     for i in range(len(heavy_heavy_atoms)):
    #         for j in range(len(heavy_heavy_atoms[i][0])):
    #             general_axes.append([heavy_heavy_atoms[i][-1], heavy_heavy_atoms[i][0][j]])
    #
    #     general_axes = [sorted(ax) for ax in general_axes]
    #     print(general_axes)
    #     # unique axes in sorted array
    #     unique_axes  = np.unique(np.array(general_axes), axis=0)
    #     # extracts both double bond and 1.5 bond from linear TS
    #     axes_ofInterest = []
    #     counted_axes = Counter([tuple(i) for i in general_axes])
    #     for key in counted_axes.keys():
    #         if counted_axes[key] < 2:
    #             axes_ofInterest.append(list(key))
    #     print(axes_ofInterest)
    #     #
    #     axes_toDelete = []
    #     for ax in axes_ofInterest:
    #         if ax[1] in gauss_bond_graph_woTSthresh[ax[0]-1]:
    #             axes_toDelete.append(ax)
    #     axes_toDelete = np.array(axes_toDelete)
    #     #
    #     axes = [ax for ax in unique_axes if ax not in axes_toDelete]
    #     axes = [list(ax) for ax in axes]
    #
    #     rotor_axes = axes

    for ax in rotor_axes:
        for i in range(len(ax)):
            ax[i] -= 1

    return rotor_axes

# -- Find the rotor dihedrals given the torsions found and the original z-matrix -- #
#Method 1 (original), groups axes, not accurate
#Not used! Need to modify print_results function to work!!
def get_rotor_dihedrals_method_1(torsions, zmat_array):
    counter = 0
    axes = {}
    rotors = {}
    zmat = []
    rotor_rows = []

    for torsion in torsions:
        axes.setdefault(torsion[1], []).append(torsion)
    for key in axes:
        rotors[key] = axes[key][0]

    for array in zmat_array:
        if len(array)==0:
            counter+=1
        if counter==2:
            zmat.append(array)
    zmat = zmat[1:]
    for rotor in rotors.values():
        rotor_rows.append(zmat[max(rotor[:4]) + 1])
    rotor_dihedrals = [row[6] for row in rotor_rows]

    return rotor_dihedrals, rotors

#Method 2, works from bond elimination algorithm - cant make rotors a dict bc unallow repeated elements
def get_rotor_dihedrals_method_2(torsions, zmat_array, bond_graph, bond_graph_woTSthresh=[]):
    counter = 0
    axes = []
    rotors = []
    zmat = []
    rotor_rows = []

    rotor_axes = get_all_rotors(bond_graph, bond_graph_woTSthresh)
    for array in zmat_array:
        if len(array)==0:
            counter+=1
        if counter==2:
            zmat.append(array)
    zmat = zmat[1:]

    largest_atomNums = []
    for torsion in torsions:
        ax = sorted(torsion[1:3])
        if (ax in rotor_axes) and (ax not in axes):
            if max(torsion[:4]) not in largest_atomNums:
                rotors.append(torsion[:4])
                axes.append(ax)
            largest_atomNums.append(max(torsion[:4]))

    for rotor in rotors:
        rotor_rows.append(zmat[max(rotor) + 1])
    rotor_dihedrals = [row[6] for row in rotor_rows]

    return rotor_dihedrals, rotors

# create New_Input _Files folder and returns its path
def create_new_input_files_directory(zmat_file):
    directory=os.getcwd()
    files=os.listdir(directory)
    new_dir = os.path.join(directory, zmat_file[:-4]+'_New_Input_Files')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return directory, new_dir

# creates n copies of the original gjf based on how many rotors it has
def create_gjf_copies(new_dir, filename, rotor_dihedrals):
    for i in range(len(rotor_dihedrals)):
        #File naming!
        shutil.copy(filename, new_dir+'/'+filename.split('.')[0]+'_'+str(i)+'_'+'.'+filename.split('.')[1])

# loops through the dihedrals and the gjf copies and parses 's 36 10.0' next to the corresponding diheral, for each copy
def write_new_files(rotor_dihedrals, originalFilenameLen):
    directory=os.getcwd()
    files=os.listdir(directory)
    files = [file for file in files if file.endswith('gjf')]
    for dihedral, filename in zip(rotor_dihedrals, files):
        file = open(filename, "r")
        file_lines = file.readlines()
        for index, line in enumerate(file_lines):
            if ((' '+dihedral+' ') in line) and (len(line)<35):
                line = list(line)
                line.insert(-1, '  s 36 10.0')
                line = "".join(line)
                file_lines[index] = line
                break
        try:
            file = open(filename, "w")
        except IOError:
            print('Error: file (%s) not found!\n' % (filename))
            sys.exit()

        file.writelines(file_lines)
        file.close()
        newname=filename.split('.')[0][:originalFilenameLen]+"_"+str(dihedral)+"_rotor.gjf"
        os.rename(filename,newname)

## -- END OF TOPOLOGY FUNCTIONS -- ##


## -- UPDATE_HEADER FUNCTION -- ## (by Katie L.)

def replace_header(ts):
    #user defined job type
    keyword='rotor'
    #user defined # of processors
    nproc=str(input("How many processors?"))

    files=os.listdir(os.getcwd())
    for filename in files:
        #calls the function "replace_header" to edit the file header for the specific file type
        f=open(filename, 'r')
        count=0
        for line in f:
            if line == '\n' or line == '\r\n':
                f.close()
                break
            else:
                count = count + 1

        #determine which route we are using based on user specified input
        if keyword=='m062x':
            if ts:
                route="#opt=(calcall,tight,ts) freq m062x/cc-pvtz maxdisk=500GB int=ultrafine"
            if not ts:
                route="#opt=(calcall,tight) freq m062x/cc-pvtz maxdisk=500GB int=ultrafine"
        elif keyword=='b3lyp':
            if ts:
                route="#opt=(calcall,tight,ts) freq b3lyp/6-311++g* maxdisk=500GB int=ultrafine"
            if not ts:
                route="#opt=(calcall,tight) freq b3lyp/6-311++g* maxdisk=500GB int=ultrafine"

        elif keyword=='E':
            route="# roccsd(t,t1diag)/cc-pvdz maxdisk=500GB int=ultraFine"

        elif keyword=='E1':
            route="# roccsd(t,t1diag)/cc-pvtz maxdisk=500GB int=ultraFine"

        elif keyword=='E2':
            route="# roccsd(t)/cc-pvqz maxdisk=500GB int=ultraFine"

        elif keyword=='IRC':
            route="# irc=(maxpoints=20,recalc=5,calcfc) m062x/cc-pvtz"

        elif keyword == 'rotor':
            if not ts:
                route="#m062x/cc-pvtz opt=internal int=ultrafine nosym"
            if ts:
                route="#m062x/cc-pvtz opt=(ts,calcfc,noeig,intern,maxcyc=50) int=ultrafine nosym"

        else:
            print("This method is not included in this code.")
        title=filename.split(".")[:-1]

        #rewrites headers
        lines=open(filename,'r').readlines()
        if '\r\n' in lines:
            text=["%nprocshared="+nproc,'\r\n',"%mem=500MW",'\r\n',"%chk="+title[0]+".chk",'\r\n',route,'\r\n']+lines[count:]
        else:
            text=["%nprocshared="+nproc,'\n',"%mem=500MW",'\n',"%chk="+title[0]+".chk",'\n',route,'\n']+lines[count:]
        open(filename,'w').writelines(text)
        f.close()

## -- END OF UPDATE_HEADER FUNCTION -- ##


## -- PRINTING RESULTS -- ##

def print_results(zmat_file, zmat, rotor_dihedrals, rotors, bond_graph):
    print("\n\n"+zmat_file)
    print('Bond Graph:')
    # adjusts indexing in bond graph
    for bonds in bond_graph:
        for i in range(len(bonds)):
            bonds[i] += 1
    print(bond_graph)
    print("--------------------------------------------------------------------------------------")
    print("Found "+str(len(rotor_dihedrals))+" rotors!\n")
    for rotor in rotors:
        for i in range(len(rotor)):
            rotor[i] += 1
    for rotor, dihedral in zip(rotors, rotor_dihedrals):
        if rotor[0]>rotor[-1]:
            pass
        else:
            rotor = rotor[::-1] #reversing list
        print(rotor, ' ----> ', dihedral)
    print("\nZ-Matrix")
    print("--------------------------------------------------------------------------------------")
#     for line in zmat_array:
#         print(''.join(line))
    print(zmat)
    print("--------------------------------------------------------------------------------------")

## -- END OF PRINTING RESULTS -- ##


## -- UNUSED PRINTING FUNCTIONS! -- ##

# print geometry to screen - Unused
def print_geom(geom, comment):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    print('%i\n%s\n' % (n_atoms, comment), end='')
    for i in range(n_atoms):
        print('%-2s' % (at_types[i]), end='')
        for j in range(3):
            print(' %12.6f' % (coords[i][j]), end='')
        print('\n', end='')
    print('\n', end='')

# print bond graph to screen - Unused
def print_bond_graph(geom, bond_graph, comment):
    at_types = geom[0]
    n_atoms = len(at_types)
    print('%s\n' % (comment), end='')
    for i in range(n_atoms):
        print(' %4i %-2s -' % (i+1, at_types[i]), end='')
        for j in range(len(bond_graph[i])):
            print(' %i' % (bond_graph[i][j] + 1), end='')
        print('\n', end='')
    print('\n', end='')

# print list of bond lengths to screen - Unused
def print_bonds(geom, bonds):
    at_types = geom[0]
    n_bonds = len(bonds)
    print('%i bond(s) found (Angstrom)' % (n_bonds))
    for q in range(n_bonds):
        n1, n2  = bonds[q][0:2]
        r12 = bonds[q][2]
        nstr = '%i-%i' % (n1+1, n2+1)
        tstr = '(%s-%s) ' % (at_types[n1], at_types[n2])
        print(' %-15s  %-13s    %6.4f\n' % (nstr, tstr, r12), end='')
    print('\n', end='')

# print list of bond angles to screen - Unused
def print_angles(geom, angles):
    at_types = geom[0]
    n_angles = len(angles)
    print('%i angle(s) found (degrees)' % (n_angles))
    for q in range(n_angles):
        n1, n2, n3 = angles[q][0:3]
        a123 = angles[q][3]
        nstr = '%i-%i-%i' % (n1+1, n2+1, n3+1)
        tstr = '(%s-%s-%s) ' % (at_types[n1], at_types[n2], at_types[n3])
        print(' %-15s  %-13s   %7.3f\n' % (nstr, tstr, a123), end='')
    print('\n', end='')

## -- END OF UNUSED PRINTING FUNCTIONS -- ##

# end of program
