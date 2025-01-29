#!/bin/bash
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail


# Check: ######################################################################
#
#   Was lassen_v100_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your lassen_v100_warpx_toss3.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SRC_DIR="/usr/workspace/${USER}/lassen-toss3/src"
SW_DIR="/usr/workspace/${USER}/lassen-toss3/gpu"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}
mkdir -p ${SRC_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# tmpfs build directory: avoids issues often seen with $HOME and is faster
build_dir=$(mktemp -d)

# c-blosc (I/O compression)
if [ -d ${SRC_DIR}/c-blosc ]
then
  cd ${SRC_DIR}/c-blosc
  git fetch --prune
  git checkout v1.21.1
  cd -
else
  git clone -b v1.21.1 https://github.com/Blosc/c-blosc.git ${SRC_DIR}/c-blosc
fi
cmake -S ${SRC_DIR}/c-blosc -B ${build_dir}/c-blosc-lassen-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
cmake --build ${build_dir}/c-blosc-lassen-build --target install --parallel 10

# HDF5
if [ -d ${SRC_DIR}/hdf5 ]
then
  cd ${SRC_DIR}/hdf5
  git fetch --prune
  git checkout hdf5-1_14_1-2
  cd -
else
  git clone -b hdf5-1_14_1-2 https://github.com/HDFGroup/hdf5.git ${SRC_DIR}/hdf5
fi
cmake -S ${SRC_DIR}/hdf5 -B ${build_dir}/hdf5-lassen-build -DBUILD_TESTING=OFF -DHDF5_ENABLE_PARALLEL=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/hdf5-1.14.1.2
cmake --build ${build_dir}/hdf5-lassen-build --target install --parallel 10

# ADIOS2
if [ -d ${SRC_DIR}/adios2 ]
then
  cd ${SRC_DIR}/adios2
  git fetch --prune
  git checkout v2.8.3
  cd -
else
  git clone -b v2.8.3 https://github.com/ornladios/ADIOS2.git ${SRC_DIR}/adios2
fi
cmake -S ${SRC_DIR}/adios2 -B ${build_dir}/adios2-lassen-build -DBUILD_TESTING=OFF -DADIOS2_BUILD_EXAMPLES=OFF -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
cmake --build ${build_dir}/adios2-lassen-build --target install -j 10

# BLAS++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/blaspp ]
then
  cd ${SRC_DIR}/blaspp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git ${SRC_DIR}/blaspp
fi
cmake -S ${SRC_DIR}/blaspp -B ${build_dir}/blaspp-lassen-build -Duse_openmp=ON -Dgpu_backend=cuda -Duse_cmake_find_blas=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build ${build_dir}/blaspp-lassen-build --target install --parallel 10

# LAPACK++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/lapackpp ]
then
  cd ${SRC_DIR}/lapackpp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git ${SRC_DIR}/lapackpp
fi
CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S ${SRC_DIR}/lapackpp -B ${build_dir}/lapackpp-lassen-build -Duse_cmake_find_lapack=ON -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31 -DLAPACK_LIBRARIES=/usr/lib64/liblapack.so
cmake --build ${build_dir}/lapackpp-lassen-build --target install --parallel 10


# Python ######################################################################
#
# sometimes, the Lassen PIP Index is down
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"

python3 -m pip install --upgrade --user virtualenv
rm -rf ${SW_DIR}/venvs/warpx-lassen-toss3
python3 -m venv ${SW_DIR}/venvs/warpx-lassen-toss3
source ${SW_DIR}/venvs/warpx-lassen-toss3/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip cache purge
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
CMAKE_PREFIX_PATH=/usr/lib64:${CMAKE_PREFIX_PATH} python3 -m pip install --upgrade -Ccompile-args="-j10" -Csetup-args=-Dblas=BLAS -Csetup-args=-Dlapack=BLAS scipy
python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
CC=mpicc H5PY_SETUP_REQUIRES=0 HDF5_DIR=${SW_DIR}/hdf5-1.14.1.2 HDF5_MPI=ON python3 -m pip install --upgrade h5py --no-cache-dir --no-build-isolation --no-binary h5py
MPLLOCALFREETYPE=1 python3 -m pip install --upgrade matplotlib==3.2.2  # does not try to build freetype itself
echo "matplotlib==3.2.2" > ${build_dir}/constraints.txt
python3 -m pip install --upgrade -c ${build_dir}/constraints.txt yt

# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r ${SRC_DIR}/warpx/requirements.txt

# for ML dependencies, see install_v100_ml.sh


# remove build temporary directory
rm -rf ${build_dir}
