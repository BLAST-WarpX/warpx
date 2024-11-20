#!/bin/bash
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail


# Check: ######################################################################
#
#   Was quartz_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your quartz_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SW_DIR="/usr/workspace/${USER}/quartz"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# tmpfs build directory: avoids issues often seen with ${HOME} and is faster
build_dir=$(mktemp -d)

# c-blosc (I/O compression)
if [ -d ${HOME}/src/c-blosc ]
then
  cd ${HOME}/src/c-blosc
  git fetch --prune
  git checkout v1.21.1
  cd -
else
  git clone -b v1.21.1 https://github.com/Blosc/c-blosc.git ${HOME}/src/c-blosc
fi
cmake -S ${HOME}/src/c-blosc -B ${build_dir}/c-blosc-quartz-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
cmake --build ${build_dir}/c-blosc-quartz-build --target install --parallel 6

# ADIOS2
if [ -d ${HOME}/src/adios2 ]
then
  cd ${HOME}/src/adios2
  git fetch --prune
  git checkout v2.8.3
  cd -
else
  git clone -b v2.8.3 https://github.com/ornladios/ADIOS2.git ${HOME}/src/adios2
fi
cmake -S ${HOME}/src/adios2 -B ${build_dir}/adios2-quartz-build -DBUILD_TESTING=OFF -DADIOS2_BUILD_EXAMPLES=OFF -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
cmake --build ${build_dir}/adios2-quartz-build --target install -j 6

# BLAS++ (for PSATD+RZ)
if [ -d ${HOME}/src/blaspp ]
then
  cd ${HOME}/src/blaspp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git ${HOME}/src/blaspp
fi
cmake -S ${HOME}/src/blaspp -B ${build_dir}/blaspp-quartz-build -Duse_openmp=ON -Duse_cmake_find_blas=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build ${build_dir}/blaspp-quartz-build --target install --parallel 6

# LAPACK++ (for PSATD+RZ)
if [ -d ${HOME}/src/lapackpp ]
then
  cd ${HOME}/src/lapackpp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git ${HOME}/src/lapackpp
fi
CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S ${HOME}/src/lapackpp -B ${build_dir}/lapackpp-quartz-build -Duse_cmake_find_lapack=ON -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build ${build_dir}/lapackpp-quartz-build --target install --parallel 6


# Python ######################################################################
#
python3 -m pip install --upgrade --user virtualenv
rm -rf ${SW_DIR}/venvs/warpx-quartz
python3 -m venv ${SW_DIR}/venvs/warpx-quartz
source ${SW_DIR}/venvs/warpx-quartz/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip cache purge
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
python3 -m pip install --upgrade -r ${HOME}/src/warpx/requirements.txt

# ML dependencies
python3 -m pip install --upgrade torch


# remove build temporary directory ############################################
#
rm -rf ${build_dir}
