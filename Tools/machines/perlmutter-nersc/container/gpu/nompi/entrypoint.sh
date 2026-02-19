#!/bin/bash --login

# Hint custom install location
export WARPX_ROOT=/opt/warpx
export CMAKE_PREFIX_PATH=${WARPX_ROOT}:${CMAKE_PREFIX_PATH}
export PATH=${WARPX_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${WARPX_ROOT}/lib:${LD_LIBRARY_PATH}

# The usual HDF5 quirk for various FS and containers
export HDF5_USE_FILE_LOCKING="FALSE"

# Activate python venv
source /opt/venv/bin/activate

# Execute the provided command
exec "$@"
