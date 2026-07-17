#!/bin/bash
#SBATCH --account=<account_to_charge>
#SBATCH --job-name=warpx
#SBATCH --constraint=MI250
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --time=00:10:00

module purge

# A CrayPE environment version
module load cpe/25.09
# An architecture
module load craype-accel-amd-gfx90a craype-x86-trento
# A compiler to target the architecture
module load rocm/6.4.3
module load PrgEnv-amd
# The MPI library
module load cray-mpich/9.0.1

date
module list

export MPICH_GPU_SUPPORT_ENABLED=1

# note
# this environment setting is needed to avoid that rocFFT writes a cache in
# the home directory, which does not scale.
export ROCFFT_RTC_CACHE_PATH=/dev/null

export OMP_NUM_THREADS=1
export WARPX_NMPI_PER_NODE=8
export TOTAL_NMPI=$(( ${SLURM_JOB_NUM_NODES} * ${WARPX_NMPI_PER_NODE} ))
srun -N${SLURM_JOB_NUM_NODES} -n${TOTAL_NMPI} --ntasks-per-node=${WARPX_NMPI_PER_NODE} \
     --cpus-per-task=8 --threads-per-core=1 --gpu-bind=closest \
    ./warpx inputs > output.txt
