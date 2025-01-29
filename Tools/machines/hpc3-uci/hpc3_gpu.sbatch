#!/bin/bash -l

# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl, Victor Flores
# License: BSD-3-Clause-LBNL

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH -J WarpX
#S BATCH -A <proj>
# V100 GPU options: gpu, free-gpu, debug-gpu
#SBATCH -p free-gpu
# use all four GPUs per node
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:4
#SBATCH --cpus-per-task=10
#S BATCH --mail-type=begin,end
#S BATCH --mail-user=<your-email>@uci.edu
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx.rz
INPUTS=inputs_rz

# OpenMP threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# run
mpirun -np ${SLURM_NTASKS} bash -c "
    export CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID};
    ${EXE} ${INPUTS}" \
  > output.txt
