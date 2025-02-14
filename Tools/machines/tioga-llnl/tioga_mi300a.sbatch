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
#SBATCH -p mi300a
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx
INPUTS=inputs

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# GPU-aware MPI optimizations
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# APUs share memory with the host:
# Do NOT pre-allocate a large heap in AMReX
APU_SHARED_MEMORY="amrex.the_arena_init_size=1"

# MPI parallel processes
srun \
  ${EXE} ${INPUTS} \
  ${GPU_AWARE_MPI} ${APU_SHARED_MEMORY} \
  > output.txt
