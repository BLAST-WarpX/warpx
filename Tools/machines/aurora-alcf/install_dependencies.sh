#!/bin/bash
#
# Copyright 2025 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl, Roelof Groenewald
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail

# Check: ######################################################################
#
#   Was aurora_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your aurora_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi

# Remove old dependencies #####################################################
#
SW_DIR="/home/${USER}/sw/aurora/gpu"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true

# General extra dependencies ##################################################
#

# BLAS++ (for PSATD+RZ)
if [ -d $HOME/src/blaspp ]
then
  cd $HOME/src/blaspp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git $HOME/src/blaspp
fi
rm -rf $HOME/src/blaspp-aurora-gpu-build
CXX=icpx CXXFLAGS="-qmkl" cmake -S $HOME/src/blaspp -B $HOME/src/blaspp-aurora-gpu-build -Duse_openmp=OFF -Dgpu_backend=sycl -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31 -DCMAKE_EXE_LINKER_FLAGS="-qmkl"
cmake --build $HOME/src/blaspp-aurora-gpu-build --target install --parallel 16
rm -rf $HOME/src/blaspp-aurora-gpu-build

# LAPACK++ (for PSATD+RZ)
if [ -d $HOME/src/lapackpp ]
then
  cd $HOME/src/lapackpp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git $HOME/src/lapackpp
fi
rm -rf $HOME/src/lapackpp-aurora-gpu-build
CXX=icpx CXXFLAGS="-DLAPACK_FORTRAN_ADD_ -qmkl" cmake -S $HOME/src/lapackpp -B $HOME/src/lapackpp-aurora-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31 -DCMAKE_EXE_LINKER_FLAGS="-qmkl"
cmake --build $HOME/src/lapackpp-aurora-gpu-build --target install --parallel 16
rm -rf $HOME/src/lapackpp-aurora-gpu-build

# Python ######################################################################
#
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
rm -rf ${SW_DIR}/venvs/warpx-aurora
python3 -m venv ${SW_DIR}/venvs/warpx-aurora
source ${SW_DIR}/venvs/warpx-aurora/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools[core]
python3 -m pip install --upgrade "cython>=3.0"
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade h5py
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
# Is next line OK/needed for Aurora?
MPICC="mpicxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \\\"-device pvc\\\" -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r $HOME/src/warpx/requirements.txt
# optional: for libEnsemble
#python3 -m pip install -r $HOME/src/warpx-aurora/Tools/LibEnsemble/requirements.txt
# optional: for optimas (based on libEnsemble & ax->botorch->gpytorch->pytorch)
#python3 -m pip install --upgrade torch  # should get from frameworks module on Aurora
#python3 -m pip install -r $HOME/src/warpx/Tools/optimas/requirements.txt
