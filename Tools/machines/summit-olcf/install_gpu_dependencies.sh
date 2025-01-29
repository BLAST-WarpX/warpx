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
#   Was perlmutter_gpu_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your summit_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Check $proj variable is correct and has a corresponding PROJWORK directory ##
#
if [ ! -d "${PROJWORK}/${proj}/" ]
then
    echo "WARNING: The directory $PROJWORK/$proj/ does not exist!"
    echo "Is the \$proj environment variable of value \"$proj\" correctly set? "
    echo "Please edit line 2 of your summit_warpx.profile file to continue!"
    exit
fi


# Check $proj variable is correct and has a corresponding Software directory ##
#
if [ ! -d "/ccs/proj/${proj}/" ]
then
    echo "WARNING: The directory /ccs/proj/$proj/ does not exist!"
    echo "Is the \$proj environment variable of value \"$proj\" correctly set? "
    echo "Please edit line 2 of your summit_warpx.profile file to continue!"
    exit
fi


# Remove old dependencies #####################################################
#
SW_DIR="/ccs/proj/${proj}/${USER}/sw/summit/gpu/"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# tmpfs build directory: avoids issues often seen with $HOME and is faster
build_dir=$(mktemp -d)

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
cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-summit-build -Duse_openmp=ON -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build ${build_dir}/blaspp-summit-build --target install --parallel 10

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
cmake -S $HOME/src/lapackpp -B ${build_dir}/lapackpp-summit-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build ${build_dir}/lapackpp-summit-build --target install --parallel 10

# remove build temporary directory
rm -rf ${build_dir}


# Python ######################################################################
#
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
rm -rf ${SW_DIR}/venvs/warpx-summit
python3 -m venv ${SW_DIR}/venvs/warpx-summit
source ${SW_DIR}/venvs/warpx-summit/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade -Ccompile-args="-j10" scipy
python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib==3.2.2  # does not try to build freetype itself
python3 -m pip install --upgrade yt

# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r $HOME/src/warpx/requirements.txt

# for ML dependencies, see install_gpu_ml.sh
