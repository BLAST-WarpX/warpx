#!/bin/bash -l

# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl, Andrei Berceanu
# License: BSD-3-Clause-LBNL

#SBATCH --account=<proj>
#SBATCH --partition=qgpu
#SBATCH --time=00:10:00
#SBATCH --job-name=WarpX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=single:1

#SBATCH --mail-type=ALL
# change me!
#SBATCH --mail-user=someone@example.com
#SBATCH --chdir=/scratch/project/<proj>/it4i-<user>/runs/warpx

#SBATCH -o stdout_%j
#SBATCH -e stderr_%j

# set user rights to u=rwx;g=r-x;o=---
umask 0027

# OpenMP threads per MPI rank
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx.rz
INPUTS=./inputs_rz

# run
srun -K1 ${EXE} ${INPUTS}
