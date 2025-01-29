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
SW_DIR="${SHAREDHOMEDIR}/sw/adastra/gpu"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# BLAS++ (for PSATD+RZ)
if [ -d $SHAREDHOMEDIR/src/blaspp ]
then
  cd $SHAREDHOMEDIR/src/blaspp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git $SHAREDHOMEDIR/src/blaspp
fi
rm -rf $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build
CXX=$(which CC) cmake -S $SHAREDHOMEDIR/src/blaspp -B $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build -Duse_openmp=OFF -Dgpu_backend=hip -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build --target install --parallel 16
rm -rf $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build

# LAPACK++ (for PSATD+RZ)
if [ -d $SHAREDHOMEDIR/src/lapackpp ]
then
  cd $SHAREDHOMEDIR/src/lapackpp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git $SHAREDHOMEDIR/src/lapackpp
fi
rm -rf $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build
CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $SHAREDHOMEDIR/src/lapackpp -B $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build --target install --parallel 16
rm -rf $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build

# c-blosc (I/O compression, for OpenPMD)
if [ -d $SHAREDHOMEDIR/src/c-blosc ]
then
  # git repository is already there
  :
else
  git clone -b v1.21.1 https://github.com/Blosc/c-blosc.git $SHAREDHOMEDIR/src/c-blosc
fi
rm -rf $SHAREDHOMEDIR/src/c-blosc-ad-build
cmake -S $SHAREDHOMEDIR/src/c-blosc -B $SHAREDHOMEDIR/src/c-blosc-ad-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
cmake --build $SHAREDHOMEDIR/src/c-blosc-ad-build --target install --parallel 16
rm -rf $SHAREDHOMEDIR/src/c-blosc-ad-build

# ADIOS2 v. 2.8.3 (for OpenPMD)
if [ -d $SHAREDHOMEDIR/src/adios2 ]
then
  # git repository is already there
  :
else
  git clone -b v2.8.3 https://github.com/ornladios/ADIOS2.git $SHAREDHOMEDIR/src/adios2
fi
rm -rf $SHAREDHOMEDIR/src/adios2-ad-build
cmake -S $SHAREDHOMEDIR/src/adios2 -B $SHAREDHOMEDIR/src/adios2-ad-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
cmake --build $SHAREDHOMEDIR/src/adios2-ad-build --target install -j 16
rm -rf $SHAREDHOMEDIR/src/adios2-ad-build


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
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
MPICC="cc -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r $SHAREDHOMEDIR/src/warpx/requirements.txt
# optional: for libEnsemble
python3 -m pip install -r $SHAREDHOMEDIR/src/warpx/Tools/LibEnsemble/requirements.txt
# optional: for optimas (based on libEnsemble & ax->botorch->gpytorch->pytorch)
#python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.4.2
#python3 -m pip install -r $SHAREDHOMEDIR/src/warpx/Tools/optimas/requirements.txt
