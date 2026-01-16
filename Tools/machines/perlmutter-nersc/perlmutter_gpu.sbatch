#!/bin/bash -l

# Copyright 2021-2023 Axel Huebl, Kevin Gott
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
# A100 40GB (most nodes)
#SBATCH -C gpu
# A100 80GB (256 nodes)
#S BATCH -C gpu&hbm80g
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
# ideally single:1, but NERSC cgroups issue
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# PICMI Python script or executable?
USE_PICMI=true

if [[ "${USE_PICMI}" = true ]]
then
    EXE=python3
    INPUTS=PICMI_script.py
    #   for GPU-aware MPI in PICMI, set
    #   Simulation(..., warpx_amrex_use_gpu_aware_mpi=True)
    GPU_AWARE_MPI=""
else
    EXE=./warpx
    INPUTS=inputs.in
    GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"
fi

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export OMP_NUM_THREADS=16

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun --cpu-bind=cores bash -c "
    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" \
  > output.txt
