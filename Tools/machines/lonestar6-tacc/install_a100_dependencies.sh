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
#   Was lonestar6_warpx_a100.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your lonestar6_warpx_a100.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SW_DIR="${WORK}/sw/lonestar6/sw/lonestar6/a100"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# tmpfs build directory: avoids issues often seen with $HOME and is faster
build_dir=$(mktemp -d)

# c-blosc (I/O compression)
if [ -d $HOME/src/c-blosc ]
then
  cd $HOME/src/c-blosc
  git fetch --prune
  git checkout v1.21.1
  cd -
else
  git clone -b v1.21.1 https://github.com/Blosc/c-blosc.git $HOME/src/c-blosc
fi
rm -rf $HOME/src/c-blosc-a100-build
cmake -S $HOME/src/c-blosc -B ${build_dir}/c-blosc-a100-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
cmake --build ${build_dir}/c-blosc-a100-build --target install --parallel 16
rm -rf ${build_dir}/c-blosc-a100-build

# ADIOS2
if [ -d $HOME/src/adios2 ]
then
  cd $HOME/src/adios2
  git fetch --prune
  git checkout v2.8.3
  cd -
else
  git clone -b v2.8.3 https://github.com/ornladios/ADIOS2.git $HOME/src/adios2
fi
rm -rf $HOME/src/adios2-a100-build
cmake -S $HOME/src/adios2 -B ${build_dir}/adios2-a100-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
cmake --build ${build_dir}/adios2-a100-build --target install -j 16
rm -rf ${build_dir}/adios2-a100-build

# BLAS++ (for PSATD+RZ)
if [ -d $HOME/src/blaspp ]
then
  cd $HOME/src/blaspp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git $HOME/src/blaspp
fi
rm -rf $HOME/src/blaspp-a100-build
cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-a100-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build ${build_dir}/blaspp-a100-build --target install --parallel 16
rm -rf ${build_dir}/blaspp-a100-build

# LAPACK++ (for PSATD+RZ)
if [ -d $HOME/src/lapackpp ]
then
  cd $HOME/src/lapackpp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git $HOME/src/lapackpp
fi
rm -rf $HOME/src/lapackpp-a100-build
CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B ${build_dir}/lapackpp-a100-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build ${build_dir}/lapackpp-a100-build --target install --parallel 16
rm -rf ${build_dir}/lapackpp-a100-build

# heFFTe
if [ -d $HOME/src/heffte ]
then
  cd $HOME/src/heffte
  git fetch --prune
  git checkout v2.4.0
  cd -
else
  git clone -b v2.4.0 https://github.com/icl-utk-edu/heffte.git ${HOME}/src/heffte
fi
rm -rf ${HOME}/src/heffte-a100-build
cmake \
    -S ${HOME}/src/heffte               \
    -B ${build_dir}/heffte-a100-build \
    -DBUILD_SHARED_LIBS=ON              \
    -DCMAKE_BUILD_TYPE=Release          \
    -DCMAKE_CXX_STANDARD=17             \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON  \
    -DCMAKE_INSTALL_PREFIX=${SW_DIR}/heffte-2.4.0  \
    -DHeffte_DISABLE_GPU_AWARE_MPI=OFF  \
    -DHeffte_ENABLE_AVX=OFF             \
    -DHeffte_ENABLE_AVX512=OFF          \
    -DHeffte_ENABLE_FFTW=OFF            \
    -DHeffte_ENABLE_CUDA=ON             \
    -DHeffte_ENABLE_ROCM=OFF            \
    -DHeffte_ENABLE_ONEAPI=OFF          \
    -DHeffte_ENABLE_MKL=OFF             \
    -DHeffte_ENABLE_DOXYGEN=OFF         \
    -DHeffte_SEQUENTIAL_TESTING=OFF     \
    -DHeffte_ENABLE_TESTING=OFF         \
    -DHeffte_ENABLE_TRACING=OFF         \
    -DHeffte_ENABLE_PYTHON=OFF          \
    -DHeffte_ENABLE_FORTRAN=OFF         \
    -DHeffte_ENABLE_SWIG=OFF            \
    -DHeffte_ENABLE_MAGMA=OFF
cmake --build ${build_dir}/heffte-a100-build --target install --parallel 16
rm -rf ${build_dir}/heffte-a100-build


# Python ######################################################################
#
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
rm -rf ${SW_DIR}/venvs/warpx-a100
python3 -m venv ${SW_DIR}/venvs/warpx-a100
source ${SW_DIR}/venvs/warpx-a100/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
# install or update WarpX dependencies
python3 -m pip install --upgrade -r $HOME/src/warpx/requirements.txt
#python3 -m pip install --upgrade cupy-cuda12x  # CUDA 12 compatible wheel
# optimas (based on libEnsemble & ax->botorch->gpytorch->pytorch)
#python3 -m pip install --upgrade torch  # CUDA 12 compatible wheel
#python3 -m pip install --upgrade optimas[all]


# remove build temporary directory
rm -rf ${build_dir}
