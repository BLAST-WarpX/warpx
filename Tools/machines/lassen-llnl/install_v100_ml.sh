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
#   Was lassen_v100_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your lassen_v100_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SRC_DIR="/usr/workspace/${USER}/lassen/src"
mkdir -p ${SRC_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qqq -y torch 2>/dev/null || true


# Python ML ###################################################################
#
# for basic python dependencies, see install_v100_dependencies.sh

# optional: for libEnsemble - WIP: issues with nlopt
# python3 -m pip install -r ${SRC_DIR}/warpx/Tools/LibEnsemble/requirements.txt

# optional: for pytorch
if [ -d ${SRC_DIR}/pytorch ]
then
  cd ${SRC_DIR}/pytorch
  git fetch
  git checkout .
  git checkout v2.0.1
  git submodule update --init --recursive
  cd -
else
  git clone -b v2.0.1 --recurse-submodules https://github.com/pytorch/pytorch.git ${SRC_DIR}/pytorch
fi
cd ${SRC_DIR}/pytorch
rm -rf build

# see https://github.com/pytorch/pytorch/issues/97497#issuecomment-1499069641
#     https://github.com/pytorch/pytorch/pull/98511
wget -q -O - https://github.com/pytorch/pytorch/pull/98511.patch | git apply

python3 -m pip install -r requirements.txt

# see https://github.com/pytorch/pytorch/issues/108984#issuecomment-1712938737
LDFLAGS="-L${CUDA_HOME}/nvidia/targets/ppc64le-linux/lib/" \
USE_CUDA=1 BLAS=OpenBLAS MAX_JOBS=64 ATEN_AVX512_256=OFF BUILD_TEST=0 python3 setup.py develop
#   (optional) If using torch.compile with inductor/triton, install the matching version of triton
#make triton
rm -rf build
cd -

# optional: optimas dependencies (based on libEnsemble & ax->botorch->gpytorch->pytorch)
# TODO: scikit-learn needs a BLAS hint
#   commented because scikit-learn et al. compile > 2 hrs
#   please run manually on a login node if needed
#python3 -m pip install -r ${SRC_DIR}/warpx/Tools/optimas/requirements.txt
