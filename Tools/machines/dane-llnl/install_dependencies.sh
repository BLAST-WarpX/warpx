#!/bin/bash
#
# Copyright 2024 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl, David Grote
# License: BSD-3-Clause-LBNL
#
# This must be run in batch since it is compute intensive.
# Also, the dane_warpx.profile script must already have been sourced.
# Run like this:
#    runNode path/to/warpx/Tools/machines/dane-llnl/install_dependencies.sh

# Exit on first error encountered #############################################
#
set -eu -o pipefail

# Make sure the script is running in batch
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: this script must be run as a SLURM batch job." >&2
    exit 1
fi

# Check: ######################################################################
#
#   Was dane_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your dane_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi

# Make sure that a virtual environment is not already activated
if declare -F deactivate &>/dev/null; then
  deactivate
fi

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true

# clean out caches, e.g., depending on old system modules
python3 -m pip cache purge || true

# Setup the directories where the packages will be installed
if [ -z "${WARPX_SW_DIR+x}" ]; then
    WARPX_SW_DIR="/usr/workspace/${USER}/dane"
fi
rm -rf ${WARPX_SW_DIR}/install
mkdir -p ${WARPX_SW_DIR}/install

# General extra dependencies ##################################################
#

# Use all available cpus on a batch node
PARALLEL=112

# tmpfs build directory: avoids issues often seen with ${HOME} and is faster
build_dir=$(mktemp -d)

# c-blosc (I/O compression)
if [ -d ${WARPX_SW_DIR}/src/c-blosc ]
then
  cd ${WARPX_SW_DIR}/src/c-blosc
  git fetch --prune
  git checkout v1.21.6
  cd -
else
  git clone -b v1.21.6 https://github.com/Blosc/c-blosc.git ${WARPX_SW_DIR}/src/c-blosc
fi
cmake -S ${WARPX_SW_DIR}/src/c-blosc -B ${build_dir}/c-blosc-dane-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${WARPX_SW_DIR}/install/c-blosc-1.21.6
cmake --build ${build_dir}/c-blosc-dane-build --target install --parallel ${PARALLEL}

# ADIOS2
if [ -d ${WARPX_SW_DIR}/src/adios2 ]
then
  cd ${WARPX_SW_DIR}/src/adios2
  git fetch --prune
  git checkout v2.10.2
  cd -
else
  git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git ${WARPX_SW_DIR}/src/adios2
fi
cmake -S ${WARPX_SW_DIR}/src/adios2 -B ${build_dir}/adios2-dane-build -DBUILD_TESTING=OFF -DADIOS2_BUILD_EXAMPLES=OFF -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF -DADIOS2_USE_ZeroMQ=OFF -DADIOS2_USE_HDF5=OFF -DCMAKE_INSTALL_PREFIX=${WARPX_SW_DIR}/install/adios2-2.10.2
cmake --build ${build_dir}/adios2-dane-build --target install --parallel ${PARALLEL}

# BLAS++ (for PSATD+RZ)
if [ -d ${WARPX_SW_DIR}/src/blaspp ]
then
  cd ${WARPX_SW_DIR}/src/blaspp
  git fetch --prune
  git checkout v2024.10.26
  cd -
else
  git clone -b v2024.10.26 https://github.com/icl-utk-edu/blaspp.git ${WARPX_SW_DIR}/src/blaspp
fi
cmake -S ${WARPX_SW_DIR}/src/blaspp -B ${build_dir}/blaspp-dane-build -Duse_openmp=ON -Duse_cmake_find_blas=ON -DCMAKE_CXX_STANDARD=20 -DCMAKE_INSTALL_PREFIX=${WARPX_SW_DIR}/install/blaspp-2024.10.26
cmake --build ${build_dir}/blaspp-dane-build --target install --parallel ${PARALLEL}

# LAPACK++ (for PSATD+RZ)
if [ -d ${WARPX_SW_DIR}/src/lapackpp ]
then
  cd ${WARPX_SW_DIR}/src/lapackpp
  git fetch --prune
  git checkout v2024.10.26
  cd -
else
  git clone -b v2024.10.26 https://github.com/icl-utk-edu/lapackpp.git ${WARPX_SW_DIR}/src/lapackpp
fi
CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S ${WARPX_SW_DIR}/src/lapackpp -B ${build_dir}/lapackpp-dane-build -Duse_cmake_find_lapack=ON -DCMAKE_CXX_STANDARD=20 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${WARPX_SW_DIR}/install/lapackpp-2024.10.26
cmake --build ${build_dir}/lapackpp-dane-build --target install --parallel ${PARALLEL}

if [[ "${PETSC_DIR:-}" == "${WARPX_SW_DIR}/src/petsc" ]]; then
  # PETSC
  # Only build it if PETSC_DIR points to this work space
  if [ -d ${WARPX_SW_DIR}/src/petsc ]
  then
    cd ${WARPX_SW_DIR}/src/petsc
    git fetch --prune
    git checkout v3.24.0
  else
    cd ${WARPX_SW_DIR}/src
    git clone -b v3.24.0 https://gitlab.com/petsc/petsc.git petsc
    cd ${WARPX_SW_DIR}/src/petsc
  fi
  ./configure --with-batch --with-fortran-bindings=no --with-x=no --with-cc=mpicc --with-fc=mpif90 --with-cxx=mpicxx COPTFLAGS="-O2" FOPTFLAGS="-O2" CXXOPTFLAGS="-O2" --with-shared-libraries --with-debugging=0 --download-hypre --download-superlu --download-superlu_dist --download-parmetis --download-metis --with-cxx-dialect=C++20
  make -j ${PARALLEL} all
  cd -
fi

# Python ######################################################################
#
# Create a virtual environment and install the Python packages there.
rm -rf ${WARPX_SW_DIR}/venvs/warpx-dane
python3 -m venv ${WARPX_SW_DIR}/venvs/warpx-dane
source ${WARPX_SW_DIR}/venvs/warpx-dane/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools[core]
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt

# install or update WarpX dependencies such as picmistandard
SCRIPT_PATH="$(realpath ${BASH_SOURCE[0]})"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
python3 -m pip install --upgrade -r ${SCRIPT_DIR}/../../../requirements.txt

# ML dependencies
python3 -m pip install --upgrade torch


# remove build temporary directory ############################################
#
rm -rf ${build_dir}
