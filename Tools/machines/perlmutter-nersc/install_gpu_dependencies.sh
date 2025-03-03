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
#   Was perlmutter_gpu_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your perlmutter_gpu_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Check $proj variable is correct and has a corresponding CFS directory #######
#
if [ ! -d "${CFS}/${proj%_g}/" ]
then
    echo "WARNING: The directory ${CFS}/${proj%_g}/ does not exist!"
    echo "Is the \$proj environment variable of value \"$proj\" correctly set? "
    echo "Please edit line 2 of your perlmutter_gpu_warpx.profile file to continue!"
    exit
fi


# Remove old dependencies #####################################################
#
SW_DIR="${PSCRATCH}/storage/sw/warpx/perlmutter/gpu"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# build parallelism
PARALLEL=16

# tmpfs build directory: avoids issues often seen with $HOME and is faster
build_dir=$(mktemp -d)

# CCache
curl -Lo ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz
tar -xf ccache.tar.xz
mv ccache-4.10.2-linux-x86_64 ${SW_DIR}/ccache-4.10.2
rm -rf ccache.tar.xz

# Boost (QED tables)
rm -rf $HOME/src/boost-temp
mkdir -p $HOME/src/boost-temp
curl -Lo $HOME/src/boost-temp/boost.tar.gz https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz
tar -xzf $HOME/src/boost-temp/boost.tar.gz -C $HOME/src/boost-temp
cd $HOME/src/boost-temp/boost_1_82_0
./bootstrap.sh --with-libraries=math --prefix=${SW_DIR}/boost-1.82.0
./b2 cxxflags="-std=c++17" install -j ${PARALLEL}
cd -
rm -rf $HOME/src/boost-temp

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
rm -rf $HOME/src/c-blosc-pm-gpu-build
cmake -S $HOME/src/c-blosc -B ${build_dir}/c-blosc-pm-gpu-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
cmake --build ${build_dir}/c-blosc-pm-gpu-build --target install --parallel ${PARALLEL}
rm -rf ${build_dir}/c-blosc-pm-gpu-build

# ADIOS2
if [ -d $HOME/src/adios2 ]
then
  cd $HOME/src/adios2
  git fetch --prune
  git checkout v2.10.2
  cd -
else
  git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git $HOME/src/adios2
fi
rm -rf $HOME/src/adios2-pm-gpu-build
cmake -S $HOME/src/adios2 -B ${build_dir}/adios2-pm-gpu-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.10.2
cmake --build ${build_dir}/adios2-pm-gpu-build --target install -j ${PARALLEL}
rm -rf ${build_dir}/adios2-pm-gpu-build

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
rm -rf $HOME/src/blaspp-pm-gpu-build
CXX=$(which CC) cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-pm-gpu-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build ${build_dir}/blaspp-pm-gpu-build --target install --parallel ${PARALLEL}
rm -rf ${build_dir}/blaspp-pm-gpu-build

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
rm -rf $HOME/src/lapackpp-pm-gpu-build
CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B ${build_dir}/lapackpp-pm-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build ${build_dir}/lapackpp-pm-gpu-build --target install --parallel ${PARALLEL}
rm -rf ${build_dir}/lapackpp-pm-gpu-build

# Python ######################################################################
#
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
rm -rf ${SW_DIR}/venvs/warpx-gpu
python3 -m venv ${SW_DIR}/venvs/warpx-gpu
source ${SW_DIR}/venvs/warpx-gpu/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
MPICC="cc -target-accel=nvidia80 -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
# install or update WarpX dependencies
python3 -m pip install --upgrade -r $HOME/src/warpx/requirements.txt
python3 -m pip install --upgrade cupy-cuda12x  # CUDA 12 compatible wheel
# optimas (based on libEnsemble & ax->botorch->gpytorch->pytorch)
python3 -m pip install --upgrade torch  # CUDA 12 compatible wheel
python3 -m pip install --upgrade optimas[all]
python3 -m pip install --upgrade lasy

# remove build temporary directory
rm -rf ${build_dir}
