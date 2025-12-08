#!/bin/bash -l

# Copyright 2025 Peter Kicsiny
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

#SBATCH --partition=milano
#SBATCH --account=facet
#SBATCH -N 2 # --nodes= number of nodes
#SBATCH -n 2 # --ntasks= number of MPI processes or tasks in total
#SBATCH -c 4 # --cpus-per-task= number of CPU cores (threads) per task
#SBATCH -t 00:10:00 # --time= walltime
#SBATCH -J WarpX # --job-name= name of job
#SBATCH --qos=preemptable # queue
#SBATCH -o WarpX.o%j # --output= stdout
#SBATCH -e WarpX.e%j # --error= stderr

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx.3d.MPI.OMP.DP.PDP.OPMD.FFT.EB.QED
INPUTS=input_simple.txt

# threads for OpenMP and threaded compressors per MPI rank
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} # -c 4 so 4 threads per MPI rank

command="module load openmpi/v4.1.6"
echo ${command}
${command}

command="mpirun -np ${SLURM_NTASKS} ${EXE} ${INPUTS}"
echo ${command}
${command}
