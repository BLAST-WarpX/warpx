#!/bin/bash -l

# Copyright 2021-2022 Axel Huebl, Kevin Gott
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:10:00
#SBATCH -N 2
#SBATCH -J WarpX
#    note: <proj> must end on _g
#SBATCH -A <proj>
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --gpus-per-node=4
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx
INPUTS=inputs_small

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# threads for OpenMP and threaded compressors per MPI rank
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# depends on https://github.com/ECP-WarpX/WarpX/issues/2009
#GPU_AWARE_MPI="amrex.the_arena_is_managed=0 amrex.use_gpu_aware_mpi=1"
GPU_AWARE_MPI=""

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun --cpu-bind=cores bash -c "
    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" \
  > output.txt
