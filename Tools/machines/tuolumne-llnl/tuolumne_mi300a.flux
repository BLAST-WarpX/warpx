#!/bin/bash

# Copyright 2025 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl, Andreas Kemp
# License: BSD-3-Clause-LBNL

### Flux directives ###
#flux: --setattr=bank=mstargt
#flux: --job-name=hemi
#flux: --nodes=16
#flux: --time-limit=360s
#flux: --queue=pbatch
#              pdebug
#flux: --exclusive
#flux: --error=WarpX.e{{id}}
#flux: --output=WarpX.o{{id}}

# Not yet tested: Transparent huge pages on CPU
# https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/pro-tips
#      --setattr=thp=always

# executable & inputs file or python interpreter & PICMI script here
EXE="./warpx.2d"
INPUTS="./inputs_hist_10.input"

# enviroment setup
if [[ -z "${MY_PROFILE}" ]]; then
    echo "WARNING: FORGOT TO"
    echo "   source $HOME/tuolumne_mi300a_warpx.profile"
    echo "before submission. Doing that now."

    source $HOME/tuolumne_mi300a_warpx.profile
fi

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# Not yet tested: Transparent huge pages on CPU
# https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/pro-tips
#export HSA_XNACK=1
#export HUGETLB_MORECORE=yes

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export OMP_NUM_THREADS=16

# GPU-aware MPI optimizations
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# start MPI parallel processes
NNODES=$(flux resource list -s up -no {nnodes})
flux run --exclusive --nodes=${NNODES} \
  --tasks-per-node=4 \
  ${EXE} ${INPUTS} \
  ${GPU_AWARE_MPI} \
  > output.txt
