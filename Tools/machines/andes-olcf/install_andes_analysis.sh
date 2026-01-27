#!/bin/bash

# Exit on first error encountered #############################################
#
set -eu -o pipefail


# Check: ######################################################################
#
#   Was andes_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your andes_warpx.profile file! Please edit its line 2 to continue and make sure you sourced the file first!"; return; fi


# Check $proj variable is correct and has a corresponding CFS directory #######
#
if [ ! -d "${PROJWORK}/${proj}/" ]
then
    echo "WARNING: The directory $PROJWORK/$proj/ does not exist!"
    echo "Is the \$proj environment variable of value \"$proj\" correctly set? "
    echo "Please edit line 2 of your andes_warpx.profile file to continue!"
    return
fi

# Install necessary dependencies for openpmd-api, as ADIOS2 version available on andes is too old
# tmpfs build directory: avoids issues often seen with $HOME and is faster
build_dir=$(mktemp -d)
echo "Building c-blosc"
if [ -d ${SW_DIR}/c-blosc ]
then
  cd ${SW_DIR}/c-blosc
  git fetch --prune
  git checkout v1.21.1
  cd -
else
  git clone -b v1.21.1 https://github.com/Blosc/c-blosc ${SW_DIR}/c-blosc
fi
if [ -d ${build_dir}/c-blosc-build ]
then
  rm -rf ${build_dir}/c-blosc-build
fi
CC=mpicc CXX=mpicxx cmake -S  ${SW_DIR}/c-blosc -B ${build_dir}/c-blosc-build -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
cmake --build  ${build_dir}/c-blosc-build --target install --parallel 16
rm -rf  ${build_dir}/c-blosc-build
echo "c-blosc built"

# ADIOS2
echo "Building adios2"
if [ -d ${SW_DIR}/adios2 ]
then
  cd ${SW_DIR}/adios2
  git fetch --prune
  git checkout v2.10.2
  cd -
else
  git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git ${SW_DIR}/adios2
fi
if [ -d ${build_dir}/adios2-build ]
then
  rm -rf ${build_dir}/adios2-build
fi
CC=mpicc CXX=mpicxx cmake -S ${SW_DIR}/adios2 -B ${build_dir}/adios2-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.10.2
cmake --build ${build_dir}/adios2-build --target install --parallel 16
rm -rf ${build_dir}/adios2-build
echo "adios2 built"

# Remove old dependencies #####################################################
#
if [ -z ${SW_DIR-} ]; then echo "WARNING: The 'SW_DIR' variable is not set"; return; fi
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python -m pip install --upgrade pip
python -m pip uninstall -qq -y pywarpx
python -m pip uninstall -qq -y warpx
python -m pip uninstall -qqq -y mpi4py 2>/dev/null || true
# Python ######################################################################
#
python -m pip install --upgrade virtualenv
python -m pip cache purge
python -m venv ${SW_DIR}/venvs/warpx-andes
source ${SW_DIR}/venvs/warpx-andes/bin/activate
echo "Installing python dependencies"
echo ""
python -m pip install --upgrade pip
python -m pip install --upgrade build
python -m pip install --upgrade packaging
python -m pip install --upgrade wheel
python -m pip install --upgrade setuptools
python -m pip install --upgrade "cython>=3.0"  # for latest mpi4py and everything else
python -m pip install --upgrade numpy
python -m pip install --upgrade scipy

# Extra packages for python ######################################################################
# 
python -m pip install --upgrade pandas
CC="mpicc -shared" CXX="mpicxx -shared" python -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py -v
# This installs openpmd-api with MPI support
CC="mpicc" CXX="mpicxx" openPMD_USE_MPI=ON openPMD_USE_ADIOS2=ON openPMD_USE_HDF5=ON python -m pip install openpmd-api --force-reinstall --no-binary openpmd-api -vv
python -m pip install --upgrade openpmd_viewer
python -m pip install --upgrade matplotlib
python -m pip install --upgrade yt
python -m pip install --upgrade dask
python -m pip install --upgrade pyarrow
python -m pip install --upgrade ipython
python -m pip install --upgrade numba
