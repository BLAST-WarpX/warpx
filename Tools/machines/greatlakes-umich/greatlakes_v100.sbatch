#!/bin/bash -l

# Copyright 2024 The WarpX Community
#
# Author: Axel Huebl
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -J WarpX
#SBATCH -A <proj>
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=v100:1
#SBATCH --gpu-bind=single:1
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx
INPUTS=inputs

# threads for OpenMP and threaded compressors per MPI rank
#   per node are 2x 2.4 GHz Intel Xeon Gold 6148
#   note: the system seems to only expose cores (20 per socket),
#         not hyperthreads (40 per socket)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# GPU-aware MPI optimizations
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# run WarpX
srun --cpu-bind=cores \
  ${EXE} ${INPUTS} ${GPU_AWARE_MPI} \
  > output.txt
