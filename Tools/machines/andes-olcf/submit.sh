#!/usr/bin/env bash

#SBATCH -A <project id>
#SBATCH -J job_name
#SBATCH -o job_name-%j.out
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -p batch

srun -n <ntasks> your_application
