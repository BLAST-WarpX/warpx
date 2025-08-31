#!/bin/bash -l

# Copyright 2024 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl, Joshua David Ludwig
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J WarpX
#S BATCH -A <proj>  # project name not needed yet
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# Transparent huge pages on CPU
# https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/pro-tips
#S  BATCH --thp=always

if [[ -z "${MY_PROFILE}" ]]; then
    echo "WARNING: FORGOT TO"
    echo "   source $HOME/tuolumne_mi300a_warpx.profile"
    echo "before submission. Doing that now."

    source $HOME/tuolumne_mi300a_warpx.profile
fi

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx
INPUTS=inputs

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# Transparent huge pages on CPU
# https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/pro-tips
#export HSA_XNACK=1
#export HUGETLB_MORECORE=yes

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export OMP_NUM_THREADS=16

# GPU-aware MPI optimizations
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# MPI parallel processes
NNODES=$(flux resource list -s up -no {nnodes})
srun -N ${NNODES} \
  --ntasks-per-node=4 \
  ${EXE} ${INPUTS} \
  ${GPU_AWARE_MPI} \
  > output.txt
