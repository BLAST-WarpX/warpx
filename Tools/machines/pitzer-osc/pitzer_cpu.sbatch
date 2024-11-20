#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks-per-node=6
#SBATCH --cpus-per-task=8
#SBATCH --job-name=<job_name>
#SBATCH --account=<project_id>
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# Pitzer cluster has 224 CPU nodes equipped with dual Intel Xeon 6148 (40 cores per node) and 340 CPU nodes with dual Intel Xeon 8268 (48 cores per node). https://www.osc.edu/resources/technical_support/supercomputers/pitzer

source ${HOME}/pitzer_cpu_warpx.profile
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# executable & inputs file or python interpreter & PICMI script here
EXE=${HOME}/src/warpx/build/bin/warpx.2d
INPUTS=inputs

srun --cpu-bind=cores ${EXE} ${INPUTS} >./logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.log 2>&1
