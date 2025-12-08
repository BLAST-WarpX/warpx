#!/bin/bash -l

# Copyright 2025 Peter Kicsiny
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

#SBATCH --partition=ampere
#SBATCH --account=facet
#SBATCH -N 1 # --nodes= number of nodes
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH -t 00:10:00 # --time= walltime
#SBATCH -J WarpX # --job-name= name of job
#SBATCH --qos=preemptable # queue
#SBATCH -o WarpX.o%j # --output= stdout
#SBATCH -e WarpX.e%j # --error= stderr

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx.3d.MPI.CUDA.DP.PDP.OPMD.FFT.EB.QED.GENQEDTABLES
INPUTS=input_simple.txt

command="module load openmpi/v4.1.6"
echo ${command}
${command}

command="srun ${EXE} ${INPUTS}"
echo ${command}
${command}
