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
# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qqq -y torch 2>/dev/null || true


# Python ML ###################################################################
#
# for basic python dependencies, see install_gpu_dependencies.sh

# optional: for libEnsemble - WIP: issues with nlopt
# python3 -m pip install -r $HOME/src/warpx/Tools/LibEnsemble/requirements.txt

# optional: for pytorch
if [ -d /ccs/proj/${proj}/${USER}/src/pytorch ]
then
  cd /ccs/proj/${proj}/${USER}/src/pytorch
  git fetch
  git checkout .
  git checkout v2.0.1
  git submodule update --init --recursive
  cd -
else
  git clone -b v2.0.1 --recurse-submodules https://github.com/pytorch/pytorch.git /ccs/proj/${proj}/${USER}/src/pytorch
fi
cd /ccs/proj/${proj}/${USER}/src/pytorch
rm -rf build
python3 -m pip install -r requirements.txt
#   patch to avoid compile issues
#   https://github.com/pytorch/pytorch/issues/97497#issuecomment-1499069641
#   https://github.com/pytorch/pytorch/pull/98511
wget -q -O - https://github.com/pytorch/pytorch/pull/98511.patch | git apply
USE_CUDA=1 BLAS=OpenBLAS MAX_JOBS=64 ATEN_AVX512_256=OFF BUILD_TEST=0 python3 setup.py develop
#   (optional) If using torch.compile with inductor/triton, install the matching version of triton
#make triton
rm -rf build
cd -

# optional: optimas dependencies (based on libEnsemble & ax->botorch->gpytorch->pytorch)
#   commented because scikit-learn et al. compile > 2 hrs
#   please run manually on a login node if needed
#python3 -m pip install -r $HOME/src/warpx/Tools/optimas/requirements.txt
