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
module load cpe/23.12
# An architecture
module load craype-accel-amd-gfx90a craype-x86-trento
# A compiler to target the architecture
module load PrgEnv-cray
# Some architecture related libraries and tools
module load CCE-GPU-3.0.0
module load amd-mixed/5.2.3

date
module list

export MPICH_GPU_SUPPORT_ENABLED=1

# note
# this environment setting is currently needed to work-around a
# known issue with Libfabric
#export FI_MR_CACHE_MAX_COUNT=0  # libfabric disable caching
# or, less invasive:
export FI_MR_CACHE_MONITOR=memhooks  # alternative cache monitor

# note
# On machines with similar architectures (Frontier, OLCF) these settings
# seem to prevent the following issue:
# OLCFDEV-1597: OFI Poll Failed UNDELIVERABLE Errors
# https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#olcfdev-1597-ofi-poll-failed-undeliverable-errors
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export FI_CXI_RX_MATCH_MODE=software

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
