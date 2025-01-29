#!/bin/bash
#PJM -L "node=48"
#PJM -L "rscgrp=small"
#PJM -L "elapse=0:30:00"
#PJM -s
#PJM -L "freq=2200,eco_state=2"
#PJM --mpi "max-proc-per-node=12"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004:/vol0003
#PJM --llio localtmp-size=10Gi
#PJM --llio sharedtmp-size=10Gi

export NODES=48
export MPI_RANKS=$((NODES * 12))
export OMP_NUM_THREADS=4

export EXE="./warpx"
export INPUT="i.3d"

export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

# Add HDF5 library path to LD_LIBRARY_PATH
# This is done manually to avoid calling spack during the run,
# since this would take a significant amount of time.
export LD_LIBRARY_PATH=/vol0004/apps/oss/spack-v0.19/opt/spack/linux-rhel8-a64fx/fj-4.8.1/hdf5-1.12.2-im6lxevf76cu6cbzspi4itgz3l4gncjj/lib:$LD_LIBRARY_PATH

# Broadcast WarpX executable to all the nodes
llio_transfer ${EXE}

mpiexec -stdout-proc ./output.%j/%/1000r/stdout -stderr-proc ./output.%j/%/1000r/stderr -n ${MPI_RANKS} ${EXE} ${INPUT}

llio_transfer --purge ${EXE}
