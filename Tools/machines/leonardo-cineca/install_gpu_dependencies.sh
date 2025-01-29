#!/bin/bash
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl, Marta Galbiati
# License: BSD-3-Clause-LBNL

set -eu -o pipefail


# Check: ######################################################################
#
#   Was leonardo_gpu_warpx.profile sourced and configured correctly?
#


# Remove old dependencies #####################################################
#
SW_DIR="$HOME/sw"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}


# General extra dependencies ##################################################
#

# ADIOS2
if [ -d $HOME/src/adios2 ]
then
  cd $HOME/src/adios2
  git fetch
  git checkout master
  git pull
  cd -
else
  git clone https://github.com/ornladios/ADIOS2.git $HOME/src/adios2
fi
rm -rf $HOME/src/adios2-gpu-build
cmake -S $HOME/src/adios2 -B $HOME/src/adios2-gpu-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-master
cmake --build $HOME/src/adios2-gpu-build --target install -j 16
rm -rf $HOME/src/adios2-gpu-build


# BLAS++ (for PSATD+RZ)
if [ -d $HOME/src/blaspp ]
then
  cd $HOME/src/blaspp
  git fetch
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git $HOME/src/blaspp
fi
rm -rf $HOME/src/blaspp-gpu-build
CXX=$(which g++) cmake -S $HOME/src/blaspp -B $HOME/src/blaspp-gpu-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build $HOME/src/blaspp-gpu-build --target install --parallel 16
rm -rf $HOME/src/blaspp-gpu-build


# LAPACK++ (for PSATD+RZ)
if [ -d $HOME/src/lapackpp ]
then
  cd $HOME/src/lapackpp
  git fetch
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git $HOME/src/lapackpp
fi
rm -rf $HOME/src/lapackpp-gpu-build
CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B $HOME/src/lapackpp-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build $HOME/src/lapackpp-gpu-build --target install --parallel 16
rm -rf $HOME/src/lapackpp-gpu-build


# Python ######################################################################
#
rm -rf ${SW_DIR}/venvs/warpx
python3 -m venv ${SW_DIR}/venvs/warpx
source ${SW_DIR}/venvs/warpx/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip cache purge
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
MPICC="gcc -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r $HOME/src/warpx/requirements.txt
# optional: for libEnsemble
python3 -m pip install -r $HOME/src/warpx/Tools/LibEnsemble/requirements.txt
# optional: for optimas (based on libEnsemble & ax->botorch->gpytorch->pytorch)
python3 -m pip install --upgrade torch  # CUDA 11.8 compatible wheel
python3 -m pip install -r $HOME/src/warpx/Tools/optimas/requirements.txt
