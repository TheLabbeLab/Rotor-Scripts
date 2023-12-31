import os
import shutil
from get_geometry import *

## -- FUNCTION CALLS AND MAIN BLOCKS -- ##

# looping through all gjfs and finding rotors for each one
directory=os.getcwd()
files=os.listdir(directory)
for file in files:
    if file.endswith('gjf'):
        zmat_file = file
        file_length = len(zmat_file.split(".")[0])
        ts = str(input("Is " + zmat_file + " a TS? (y/n): "))
        # if ts == 'y':
        #     ts = True
        #     bond_thresh = 1.4
        # else:
        #     ts = False
        # get original zmat array
        zmat_array, zmat = get_file_string_array(zmat_file)
        # read in z-matrix
        mol = zmat2xyz.molecule(zmat_file)
        # convert to xyz-coordinates
        xyz_array = mol.zmat2xyz()

        # read in geometry, determine bonded topology
        geom = get_geom(xyz_array)

        if ts == 'y':
            ts = True
            # calculates bond graph without changing threshold. Used for extracting rotors of non-ring-structured TS
            bond_thresh = 1.2
            bond_graph_woTSthresh, double_bonds = get_bond_graph(geom, bond_thresh)
            bond_thresh = 1.4
        else:
            ts = False
            bond_thresh = 1.2

        bond_graph, double_bonds = get_bond_graph(geom, bond_thresh)
        # calculate bond lengths, angles, and torsions
        bonds = get_bonds(geom, bond_graph)
        angles = get_angles(geom, bond_graph)
        torsions = get_torsions(geom, bond_graph)
        if ts:
            rotor_dihedrals, rotors = get_rotor_dihedrals_method_2(torsions, zmat_array, bond_graph, bond_graph_woTSthresh)
        else:
            rotor_dihedrals, rotors = get_rotor_dihedrals_method_2(torsions, zmat_array, bond_graph)

        print('double bonds (Py indexing):', double_bonds)

        # filtering out double bonds from possible rotors
        new_rotors = [[rotor, dihedral] for [rotor, dihedral] in zip(rotors, rotor_dihedrals) if rotor[1:3] not in double_bonds]
        rotors = [item[0] for item in new_rotors]
        rotor_dihedrals = [item[1] for item in new_rotors]

        current_dir, new_dir = create_new_input_files_directory(zmat_file)
        create_gjf_copies(new_dir, zmat_file, rotor_dihedrals)
        os.chdir(new_dir)
        # write new gjf file
        write_new_files(rotor_dihedrals, file_length)
        replace_header(ts)
        os.chdir(current_dir)

        # prints results
        # print_results(zmat_file, zmat, rotor_dihedrals, rotors, bond_graph)

#--------------------- Below Peice Added by Pray Shah -----------------------#

run_shell_script = """#!/bin/bash
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --job-name=rotors
#SBATCH --account=ucb273_peak1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prsh1291@colorado.edu

module purge
module load "gaussian/16_avx2"

FILES=*.gjf
for f in $FILES
do
 name=$(echo "$f" | cut -f 1 -d '.')
 g16 <$name.gjf> $name.log
  echo "Processing $name file..."
done

date
"""

for dirs in os.listdir():
    if dirs.endswith('_New_Input_Files'):
        RotorFilesPath = os.path.abspath(dirs)

RotorNewDir = []

for fileName in os.listdir(RotorFilesPath):
    os.makedirs(fileName.split('_')[-2])
    shutil.move(os.path.join(RotorFilesPath, fileName), fileName.split("_")[-2])
    ShellFilePath = os.path.join(fileName.split("_")[-2], 'run_all_gjfs_series.sh')
    f = open(ShellFilePath, 'w')
    f.writelines(run_shell_script)
    f.close()

os.rmdir(RotorFilesPath)

print("\nRotor job input files and shell scripts are printed\n")
## -- END FUNCTION CALLS AND MAIN BLOCKS -- ##
