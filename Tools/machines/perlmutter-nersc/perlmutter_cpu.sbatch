#!/bin/bash -l

# Copyright 2021-2023 WarpX
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:10:00
#SBATCH -N 2
#SBATCH -J WarpX
#SBATCH -A <proj>
#SBATCH -q regular
#SBATCH -C cpu
# 8 cores per chiplet, 2x SMP
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx
INPUTS=inputs_small

# each CPU node on Perlmutter (NERSC) has 64 hardware cores with
# 2x Hyperthreading/SMP
# https://en.wikichip.org/wiki/amd/epyc/7763
# https://www.amd.com/en/products/cpu/amd-epyc-7763
# Each CPU is made up of 8 chiplets, each sharing 32MB L3 cache.
# This will be our MPI rank assignment (2x8 is 16 ranks/node).

# threads for OpenMP and threaded compressors per MPI rank
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --cpu-bind=cores \
  ${EXE} ${INPUTS} \
  > output.txt
