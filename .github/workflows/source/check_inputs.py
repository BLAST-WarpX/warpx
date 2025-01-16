#! /usr/bin/env python3

import os
import re
import sys

# mandatory prefixes for test names
testname_prefix = ["test_1d_", "test_2d_", "test_3d_", "test_rz_"]

# collect all test names and test input filenames from CMakeLists.txt files
tests = []
# walk through all files under Examples/, including subdirectories
for dirpath, dirnames, filenames in os.walk(top="./Examples"):
    # loop over CMakeLists.txt files
    for name in [filename for filename in filenames if filename == "CMakeLists.txt"]:
        filepath = os.path.join(dirpath, name)
        # open CMakeLists.txt file
        with open(filepath) as f:
            # loop over lines of CMakeLists.txt
            for line in f:
                # strip leading whitespaces
                line = line.lstrip()
                # find lines where 'add_warpx_test' is called
                if re.match("add_warpx_test", line):
                    # strip leading whitespaces, remove end-of-line comments
                    testname = next(f).lstrip().split(" ")[0]
                    # skip lines related to other function arguments
                    # NOTE: update range call to reflect changes
                    #       in the interface of 'add_warpx_test'
                    for _ in range(2):  # skip over: dims, nprocs
                        next(f)
                    # strip leading whitespaces, remove end-of-line comments
                    testinput = next(f).lstrip().split(" ")[0]
                    # some Python input scripts are quoted
                    # to account for command-line arguments:
                    # strip initial quotation mark from string
                    if testinput.startswith('"'):
                        testinput = re.sub('"', "", testinput)
                    # extract filename from path
                    testinput = os.path.split(testinput)[1]
                    tests.append(
                        {"name": testname, "input": testinput, "path": filepath}
                    )

# check consistency of test names and test input filenames
print("\nCheck that test names and input names are correct...")
wrong_testname = False
wrong_testinput = False
for test in tests:
    testname = test["name"].rstrip()
    testinput = test["input"].rstrip()
    testpath = test["path"].rstrip()
    if not testname.startswith(tuple(testname_prefix)):
        print(f"Wrong test  name: {testname}")
        print(f"(from {testpath})")
        wrong_testname = True
    # PICMI tests
    if "picmi" in testname:
        if not testname.endswith("_picmi") and not testname.endswith("_picmi_restart"):
            print(f"Wrong test  name: {testname}")
            print(f"(from {testpath})")
            wrong_testname = True
    # restart tests
    if "restart" in testname:
        if not testname.endswith("_restart"):
            print(f"Wrong test  name: {testname}")
            print(f"(from {testpath})")
            wrong_testname = True
    # test input file names
    if (
        not testinput == f"inputs_{testname}"
        and not testinput == f"inputs_{testname}.py"
    ):
        # we may be running a base input file/script or a restart PICMI test
        if not testinput.startswith("inputs_base") and not testinput.endswith(
            "_picmi.py"
        ):
            print(f"Wrong input name: {testinput}")
            print(f"(from {testpath})")
            wrong_testinput = True

if wrong_testname:
    print(f"NOTE: Test names must start with one of {testname_prefix}.")
    print("      Test names must end with '_restart' for restart tests.")
    print("      Test names must end with '_picmi' for PICMI tests.")
if wrong_testinput:
    print("NOTE: Test input names must start with 'inputs_' followed by the test name.")
    print("      Test input names must end with '.py' for PICMI tests.")

# check that all input files in Examples/ are tested
print("\nCheck that all test input files are tested...")
missing_input = False
# walk through all files under Examples/, including subdirectories
for dirpath, dirnames, filenames in os.walk(top="./Examples"):
    # loop over files starting with "inputs_test_"
    for name in [
        filename for filename in filenames if filename.startswith("inputs_test_")
    ]:
        if name not in [test["input"] for test in tests]:
            print(f"Input not tested: {os.path.join(dirpath, name)}")
            missing_input = True

if missing_input:
    print("NOTE: All test input files must be tested.\n")
else:
    print()

if wrong_testname or wrong_testinput or missing_input:
    sys.exit("FAILED\n")
