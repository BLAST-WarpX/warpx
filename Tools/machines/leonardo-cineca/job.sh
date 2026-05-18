#!/usr/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=494000
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=<job name>
#SBATCH --gres=gpu:4
#SBATCH --err=job.err
#SBATCH --out=job.out
#SBATCH --account=<project id>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<mail>

cd /leonardo_scratch/large/userexternal/<username>/<directory>
srun /leonardo/home/userexternal/<username>/src/warpx/build_gpu/bin/warpx.2d <input file> > output.txt
