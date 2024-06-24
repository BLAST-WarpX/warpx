#!/bin/bash

# Copyright 2018-2020 Axel Huebl, David Grote, Edoardo Zoni
# Luca Fedeli, Maxence Thevenet, Remi Lehe
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script runs some of WarpX's standard regression tests, but
# without comparing the output to previously run simulations.
# This checks that:
# - The code compiles and runs without error
# - For some of the tests, a Python script checks that the results are
# physically correct.

# The tests can be influenced by environment variables:
# Use `export WARPX_CI_DIM=3` or `export WARPX_CI_DIM=2` in order to
# select only the tests that correspond to this dimension
# Use `export WARPX_TEST_ARCH=CPU` or `export WARPX_TEST_ARCH=GPU` in order
# to run the tests on CPU or GPU respectively.

set -eu -o pipefail

# Parse command line arguments: if test names are given as command line arguments,
# store them in variable tests_arg and define new command line argument to call
# regtest.py with option --tests (works also for single test)
tests_arg=$*
tests_run=${tests_arg:+--tests=${tests_arg}}

# environment options
WARPX_CI_TMP=${WARPX_CI_TMP:-""}

# Remove contents and link to a previous test directory (intentionally two arguments)
rm -rf test_dir/* test_dir
# Create a temporary test directory
if [ -z "${WARPX_CI_TMP}" ]; then
    tmp_dir=$(mktemp --help >/dev/null 2>&1 && mktemp -d -t ci-XXXXXXXXXX || mktemp -d "${TMPDIR:-/tmp}"/ci-XXXXXXXXXX)
    if [ $? -ne 0 ]; then
        echo "Cannot create temporary directory"
        exit 1
    fi
else
    tmp_dir=${WARPX_CI_TMP}
fi

# Copy WarpX into current test directory
rm -rf ${tmp_dir}/warpx
mkdir -p ${tmp_dir}/warpx
cp -r ./* ${tmp_dir}/warpx

# Link the test directory
ln -s ${tmp_dir} test_dir

# Switch to the test directory
cd test_dir
echo "cd $PWD"

# Prepare a virtual environment
rm -rf py-venv
python3 -m venv py-venv
source py-venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build packaging setuptools wheel
python3 -m pip install --upgrade cmake
python3 -m pip install --upgrade -r warpx/Regression/requirements.txt

# Clone AMReX and warpx-data
git clone https://github.com/AMReX-Codes/amrex.git
cd amrex && git checkout --detach 1f038e767011e20100af6ab0db02c69cf7ebe55c && cd -
# warpx-data contains various required data sets
git clone --depth 1 https://github.com/ECP-WarpX/warpx-data.git
# openPMD-example-datasets contains various required data sets
mkdir -p openPMD-example-datasets
cd openPMD-example-datasets
curl -sOL https://github.com/openPMD/openPMD-example-datasets/raw/4ba1d257c5b4897c0a3cd57742bb0987343a902e/example-femm-thetaMode.h5
curl -sOL https://github.com/openPMD/openPMD-example-datasets/raw/4ba1d257c5b4897c0a3cd57742bb0987343a902e/example-femm-3d.h5
cd -

# Clone the AMReX regression test utility
git clone https://github.com/AMReX-Codes/regression_testing.git
# FIXME: https://github.com/AMReX-Codes/regression_testing/issues/136
cd regression_testing && git checkout 93ddfb11456f47d6555c39388ba1a4ead61fbf4e && cd -

# Prepare regression tests
mkdir -p rt-WarpX/WarpX-benchmarks
cd warpx/Regression
echo "cd $PWD"
python3 prepare_file_ci.py
cp ci-tests.ini ../../rt-WarpX
cp -r Checksum ../../regression_testing/

# Run tests
cd ../../regression_testing/
echo "cd $PWD"
# run only tests specified in variable tests_arg (single test or multiple tests)
if [[ ! -z "${tests_arg}" ]]; then
  python3 regtest.py ../rt-WarpX/ci-tests.ini --no_update all "${tests_run}"
# run all tests (variables tests_arg and tests_run are empty)
else
  python3 regtest.py ../rt-WarpX/ci-tests.ini --no_update all
fi

# clean up python virtual environment
cd ../
echo "cd $PWD"
deactivate
