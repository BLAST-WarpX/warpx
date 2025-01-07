#!/bin/bash
#
# Copyright 2024 The WarpX Community
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
#   Was tioga_mi300a_warpx.profile sourced and configured correctly?
#   early access: not yet used!
#if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your tioga_mi300a_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SRC_DIR="/p/lustre1/${USER}/tioga/src"
SW_DIR="/p/lustre1/${USER}/tioga/warpx/mi300a"

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qqq -y torch 2>/dev/null || true


# Python ML ###################################################################
#
# for basic python dependencies, see install_mi300a_dependencies.sh

# sometimes, the Lassen PIP Index is down
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"

source ${SW_DIR}/venvs/warpx-trioga-mi300a/bin/activate

python3 -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
python3 -m pip install --upgrade scikit-learn
python3 -m pip install --upgrade "optimas[all]"
