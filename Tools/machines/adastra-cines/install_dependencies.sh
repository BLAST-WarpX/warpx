#!/bin/bash
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl, Luca Fedeli
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail


# Check: ######################################################################
#
#   Was perlmutter_gpu_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your adastra_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SW_DIR="${WORKDIR}/sw/adastra/gpu"
SRC_DIR="${WORKDIR}/src"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# define how many threads are used for compilation
PARALLEL=16

# BLAS++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/blaspp ]
then
  cd ${SRC_DIR}/blaspp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git ${SRC_DIR}/blaspp
fi
rm -rf ${SRC_DIR}/blaspp-adastra-gpu-build
cmake -S ${SRC_DIR}/blaspp -B ${SRC_DIR}/blaspp-adastra-gpu-build -Duse_openmp=OFF -Dgpu_backend=hip -DGPU_TARGETS=gfx90a  -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build ${SRC_DIR}/blaspp-adastra-gpu-build --target install --parallel ${PARALLEL}
rm -rf ${SRC_DIR}/blaspp-adastra-gpu-build

# LAPACK++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/lapackpp ]
then
  cd ${SRC_DIR}/lapackpp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git ${SRC_DIR}/lapackpp
fi
rm -rf ${SRC_DIR}/lapackpp-adastra-gpu-build
cmake -S ${SRC_DIR}/lapackpp -B ${SRC_DIR}/lapackpp-adastra-gpu-build -DGPU_TARGETS=gfx90a  -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build ${SRC_DIR}/lapackpp-adastra-gpu-build --target install --parallel ${PARALLEL}
rm -rf ${SRC_DIR}/lapackpp-adastra-gpu-build

# c-blosc2 (I/O compression, for OpenPMD)
if [ -d ${SRC_DIR}/c-blosc2 ]
then
  # git repository is already there
  :
else
  git clone -b v2.23.0 https://github.com/Blosc/c-blosc2.git ${SRC_DIR}/c-blosc2
fi
rm -rf ${SRC_DIR}/c-blosc2-ad-build
cmake -S ${SRC_DIR}/c-blosc2 -B ${SRC_DIR}/c-blosc2-ad-build -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF  -DBUILD_FUZZERS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-2.23.0
cmake --build ${SRC_DIR}/c-blosc2-ad-build --target install --parallel ${PARALLEL}
rm -rf ${SRC_DIR}/c-blosc2-ad-build

# c-blosc2 (I/O compression, for OpenPMD)
if [ -d ${SRC_DIR}/c-blosc2 ]
then
  # git repository is already there
  :
else
  git clone -b v2.23.0 https://github.com/Blosc/c-blosc2.git ${SRC_DIR}/c-blosc2
fi
rm -rf ${SRC_DIR}/c-blosc2-ad-build
cmake -S ${SRC_DIR}/c-blosc2 -B ${SRC_DIR}/c-blosc2-ad-build -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF  -DBUILD_FUZZERS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-2.23.0
cmake --build ${SRC_DIR}/c-blosc2-ad-build --target install --parallel ${PARALLEL}
rm -rf ${SRC_DIR}/c-blosc2-ad-build


# Python ######################################################################
#
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
rm -rf ${SW_DIR}/venvs/warpx-adastra
python3 -m venv ${SW_DIR}/venvs/warpx-adastra
source ${SW_DIR}/venvs/warpx-adastra/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools[core]
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
python3 -m pip install --upgrade jupyter
MPICC="cc -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
python3 -m pip install --upgrade openpmd-viewer
python3 -m pip install --upgrade adios2
# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r ${SRC_DIR}/warpx/requirements.txt
# optional: for optimas (based on libEnsemble & ax->botorch->gpytorch->pytorch)
python3 -m pip install 'optimas[all]'
# optional: for lasy
python3 -m pip install --upgrade lasy
