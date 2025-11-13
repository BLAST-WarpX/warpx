#!/usr/bin/env bash

#SBATCH -A <project id>
#SBATCH -J warpx
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH --ntasks-per-node=8
# Due to Frontier's Low-Noise Mode Layout only 7 instead of 8 cores are available per process
# https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#low-noise-mode-layout
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -N 20

# load cray libs and ROCm libs
#export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

# From the documentation:
# Each Frontier compute node consists of [1x] 64-core AMD EPYC 7A53
# "Optimized 3rd Gen EPYC" CPU (with 2 hardware threads per physical core) with
# access to 512 GB of DDR4 memory.
# Each node also contains [4x] AMD MI250X, each with 2 Graphics Compute Dies
# (GCDs) for a total of 8 GCDs per node. The programmer can think of the 8 GCDs
# as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).

# note (5-16-22 and 7-12-22)
# this environment setting is currently needed on Frontier to work-around a
# known issue with Libfabric (both in the May and June PE)
#export FI_MR_CACHE_MAX_COUNT=0  # libfabric disable caching
# or, less invasive:
export FI_MR_CACHE_MONITOR=memhooks  # alternative cache monitor

# Seen since August 2023
# OLCFDEV-1597: OFI Poll Failed UNDELIVERABLE Errors
# https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#olcfdev-1597-ofi-poll-failed-undeliverable-errors
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export FI_CXI_RX_MATCH_MODE=software

# note (9-2-22, OLCFDEV-1079)
# this environment setting is needed to avoid that rocFFT writes a cache in
# the home directory, which does not scale.
export ROCFFT_RTC_CACHE_PATH=/dev/null

export OMP_NUM_THREADS=1
export WARPX_NMPI_PER_NODE=8
export TOTAL_NMPI=$(( ${SLURM_JOB_NUM_NODES} * ${WARPX_NMPI_PER_NODE} ))
export WARPX_OUTPUT=output.txt

# For large-scale simulations, uncomment this line and manually set the lfs striping
# Incorrect striping could result in very slow file writing and simulations hanging
# lfs setstripe -c 1 -S 16M $SLURM_SUBMIT_DIR
# echo "Set the striping of folder ${SLURM_SUBMIT_DIR} to 'lfs setstripe -c 1 -S 16M'"

srun --kill-on-bad-exit=1 -N${SLURM_JOB_NUM_NODES} -n${TOTAL_NMPI} --ntasks-per-node=${WARPX_NMPI_PER_NODE} \
    ./warpx inputs > ${WARPX_OUTPUT} &

# This is a "watchdog" script that checks if the simulation is alive.
# The way the check is perfomed is by verifying that the output file
# has been modified within the past timeout_sec seconds.
# If not, it is assumed that the simulation is hanging and a stop signal is sent.
# This prevents a waste of simulation time in case the simulation starts when
# the user is not able to check it in real time.
# The check on the output file is performed every check_interval seconds.
# The idea is that check_interval is a frequent check and it prevents the watchdog script
# from keeping the node busy in case the simulation ends correctly between two checks.

# Please adjust timeout_sec and check_interval if needed
srun_pid=$!

timeout_sec=1200  # timeout: 20 min
check_interval=200  # check every 200 seconds

while kill -0 "$srun_pid" 2>/dev/null
do
    sleep ${check_interval}
    
    # Check if output file has been modified recently
    if [[ -f ${WARPX_OUTPUT} ]]; then
        file_mtime=$(stat -c %Y ${WARPX_OUTPUT})
        now_time=$(date +%s)
        diff_sec=$((now_time - file_mtime))

        if [[ ${diff_sec} -ge ${timeout_sec} ]]
        then
            echo "Job did not progress for ${timeout_sec} seconds..."
            echo "Probably hanging... Will terminate now."
            kill -15 ${srun_pid}
            echo "Sent a SIGTERM. Giving a chance to write a checkpoint"
            sleep 800
            kill -9 ${srun_pid}
            break
        fi
    fi
done

# Wait for the process to fully exit and capture its exit code
wait $srun_pid
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo 'Simulation finished successfully'
else
    echo "Simulation exited with code $exit_code"
fi
