#!/bin/bash -l

# Just increase this number of you need more nodes.
#SBATCH -N 2
#SBATCH -t 24:00:00
#SBATCH -A <allocation ID>

#SBATCH -J WarpX
#SBATCH -q pbatch
#SBATCH --qos=normal
#SBATCH --license=lustre1,lustre2
#SBATCH --export=ALL
#SBATCH -e error.txt
#SBATCH -o output.txt
# one MPI rank per half-socket (see below)
#SBATCH --tasks-per-node=2
# request all logical (virtual) cores per half-socket
#SBATCH --cpus-per-task=112


# each Dane node has 2 sockets of Intel Sapphire Rapids with 56 cores each
export WARPX_NMPI_PER_NODE=2

# each MPI rank per half-socket has 56 physical cores
#   or 112 logical (virtual) cores
# over-subscribing each physical core with 2x
#   hyperthreading led to a slight (3.5%) speedup on Cori's Intel Xeon E5-2698 v3,
#   so we do the same here
# the settings below make sure threads are close to the
#   controlling MPI rank (process) per half socket and
#   distribute equally over close-by physical cores and,
#   for N>9, also equally over close-by logical cores
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=112

EXE="<path/to/executable>"  # e.g. ./warpx

srun --cpu_bind=cores -n $(( ${SLURM_JOB_NUM_NODES} * ${WARPX_NMPI_PER_NODE} )) ${EXE} <input file>
