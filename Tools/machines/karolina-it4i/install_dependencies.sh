#!/bin/bash
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl, Andrei Berceanu
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #################################
#
set -eu -o pipefail

# Check: ##########################################################
#
# Was karolina_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then
    echo "WARNING: The 'proj' variable is not yet set in your karolina_warpx.profile file!"
    echo "Please edit its line 2 to continue!"
    return
fi

# download and activate spack
# this might take about ~ 1 hour
if [ ! -d "$WORK/spack" ]
then
    git clone -c feature.manyFiles=true -b v0.21.0 https://github.com/spack/spack.git $WORK/spack
    source $WORK/spack/share/spack/setup-env.sh
else
    # If the directory exists, checkout v0.21.0 branch
    cd $WORK/spack
    git checkout v0.21.0
    git pull origin v0.21.0
    source $WORK/spack/share/spack/setup-env.sh

    # Delete spack env if present
    spack env deactivate || true
    spack env rm -y warpx-karolina-cuda || true

    cd -
fi

# create and activate the spack environment
spack env create warpx-karolina-cuda $WORK/src/warpx/Tools/machines/karolina-it4i/spack-karolina-cuda.yaml
spack env activate warpx-karolina-cuda
spack install

# Python ##########################################################
#
python -m pip install --user --upgrade pandas
python -m pip install --user --upgrade matplotlib
# optional
#python -m pip install --user --upgrade yt

# install or update WarpX dependencies
python -m pip install --user --upgrade picmistandard==0.33.0
python -m pip install --user --upgrade lasy

# optional: for optimas (based on libEnsemble & ax->botorch->gpytorch->pytorch)
# python -m pip install --user --upgrade -r $WORK/src/warpx/Tools/optimas/requirements.txt
