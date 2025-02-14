#!/usr/bin/env python3

# This is a script that analyses the simulation results from
# the script `inputs_test_2d_pass_mpi_comm_picmi.py`.

import sys

import matplotlib

matplotlib.use("Agg")
import yt

yt.funcs.mylog.setLevel(50)
import numpy as np

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
import checksum

# this will be the name of the first plot file
fn1 = "Python_pass_mpi_comm_plt1_000010"
# second plot file
fn2 = "Python_pass_mpi_comm_plt2_000010"

test_name1 = fn1[:-10]
test_name2 = fn2[:-10]


checksum1 = checksum.Checksum(test_name1, fn1, do_fields=True, do_particles=True)

checksum2 = checksum.Checksum(test_name2, fn2, do_fields=True, do_particles=True)

rtol = 1.0e-9
atol = 1.0e-40

# Evaluate checksums against each other, adapted from
# Checksum.evaluate() method

# Dictionaries have same outer keys (levels, species)?
if checksum1.data.keys() != checksum2.data.keys():
    print("ERROR: plotfile 1 and plotfile 2 checksums have different outer keys:")
    print("Plot1: %s" % checksum1.data.keys())
    print("Plot2: %s" % checksum2.data.keys())
    sys.exit(1)

# Dictionaries have same inner keys (field and particle quantities)?
for key1 in checksum1.data.keys():
    if checksum1.data[key1].keys() != checksum2.data[key1].keys():
        print("ERROR: plotfile 1 and plotfile 2 checksums have different inner keys:")
        print("Common outer keys: %s" % checksum2.data.keys())
        print("Plotfile 1 inner keys in %s: %s" % (key1, checksum1.data[key1].keys()))
        print("Plotfile 2 inner keys in %s: %s" % (key1, checksum2.data[key1].keys()))
        sys.exit(1)

# Dictionaries have same values?
checksums_same = False
for key1 in checksum1.data.keys():
    for key2 in checksum1.data[key1].keys():
        passed = np.isclose(
            checksum2.data[key1][key2], checksum1.data[key1][key2], rtol=rtol, atol=atol
        )
        # skip over these, since they will be the same if communicators
        # have same number of procs
        if key2 in ["particle_cpu", "particle_id", "particle_position_y"]:
            continue
        if passed:
            print(
                "ERROR: plotfile 1 and plotfile 2 checksums have "
                "same values for key [%s,%s]" % (key1, key2)
            )
            print(
                "Plotfile 1: [%s,%s] %.15e" % (key1, key2, checksum1.data[key1][key2])
            )
            print(
                "Plotfile 2: [%s,%s] %.15e" % (key1, key2, checksum2.data[key1][key2])
            )
            checksums_same = True
if checksums_same:
    sys.exit(1)
