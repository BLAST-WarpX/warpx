#!/bin/bash
#
# Copyright 2024 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl, David Grote
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail


# Check: ######################################################################
#
#   Was dane_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your dane_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# The src directory should have already been created when cloning WarpX #######
#
SW_DIR="/usr/workspace/${USER}/dane"
rm -rf ${SW_DIR}/install
mkdir -p ${SW_DIR}/install

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# tmpfs build directory: avoids issues often seen with ${HOME} and is faster
build_dir=$(mktemp -d)

# c-blosc (I/O compression)
if [ -d ${SW_DIR}/src/c-blosc ]
then
  cd ${SW_DIR}/src/c-blosc
  git fetch --prune
  git checkout v1.21.6
  cd -
else
  git clone -b v1.21.6 https://github.com/Blosc/c-blosc.git ${SW_DIR}/src/c-blosc
fi
cmake -S ${SW_DIR}/src/c-blosc -B ${build_dir}/c-blosc-dane-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/install/c-blosc-1.21.6
cmake --build ${build_dir}/c-blosc-dane-build --target install --parallel 6

# ADIOS2
if [ -d ${SW_DIR}/src/adios2 ]
then
  cd ${SW_DIR}/src/adios2
  git fetch --prune
  git checkout v2.8.3
  cd -
else
  git clone -b v2.8.3 https://github.com/ornladios/ADIOS2.git ${SW_DIR}/src/adios2
fi
cmake -S ${SW_DIR}/src/adios2 -B ${build_dir}/adios2-dane-build -DBUILD_TESTING=OFF -DADIOS2_BUILD_EXAMPLES=OFF -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/install/adios2-2.8.3
cmake --build ${build_dir}/adios2-dane-build --target install -j 6

# BLAS++ (for PSATD+RZ)
if [ -d ${SW_DIR}/src/blaspp ]
then
  cd ${SW_DIR}/src/blaspp
  git fetch --prune
  git checkout v2024.10.26
  cd -
else
  git clone -b v2024.10.26 https://github.com/icl-utk-edu/blaspp.git ${SW_DIR}/src/blaspp
fi
cmake -S ${SW_DIR}/src/blaspp -B ${build_dir}/blaspp-dane-build -Duse_openmp=ON -Duse_cmake_find_blas=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/install/blaspp-2024.10.26
cmake --build ${build_dir}/blaspp-dane-build --target install --parallel 6

# LAPACK++ (for PSATD+RZ)
if [ -d ${SW_DIR}/src/lapackpp ]
then
  cd ${SW_DIR}/src/lapackpp
  git fetch --prune
  git checkout v2024.10.26
  cd -
else
  git clone -b v2024.10.26 https://github.com/icl-utk-edu/lapackpp.git ${SW_DIR}/src/lapackpp
fi
CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S ${SW_DIR}/src/lapackpp -B ${build_dir}/lapackpp-dane-build -Duse_cmake_find_lapack=ON -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/install/lapackpp-2024.10.26
cmake --build ${build_dir}/lapackpp-dane-build --target install --parallel 6


# Python ######################################################################
#
python3 -m pip install --upgrade --user virtualenv
rm -rf ${SW_DIR}/venvs/warpx-dane
python3 -m venv ${SW_DIR}/venvs/warpx-dane
source ${SW_DIR}/venvs/warpx-dane/bin/activate
python3 -m pip install --upgrade pip
#python3 -m pip cache purge
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt

# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r ${SW_DIR}/src/warpx/requirements.txt

# ML dependencies
python3 -m pip install --upgrade torch


# remove build temporary directory ############################################
#
rm -rf ${build_dir}
