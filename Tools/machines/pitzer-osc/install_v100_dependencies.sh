#!/bin/bash
#
# Copyright 2024 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Zhongwei Wang
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail

# Check: ######################################################################
#
#   Was pitzer_v100_warpx.profile sourced and configured correctly?
if [ -z ${proj-} ]; then
  echo "WARNING: The 'proj' variable is not yet set in your pitzer_v100_warpx.profile file! Please edit its line 2 to continue!"
  exit 1
fi

# Remove old dependencies #####################################################
#
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true

# General extra dependencies ##################################################
#
SRC_DIR="${HOME}/src"
build_dir=$(mktemp -d)

# boost (for QED table generation support)
cd ${SRC_DIR}
wget https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz
tar -xzvf boost_1_82_0.tar.gz
rm -rf boost_1_82_0.tar.gz
cd -

cd ${SRC_DIR}/boost_1_82_0
./bootstrap.sh --prefix=${SW_DIR}/boost-1.82.0
./b2 install
cd -

# BLAS++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/blaspp ]; then
  cd ${SRC_DIR}/blaspp
  git fetch
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git ${SRC_DIR}/blaspp
fi
rm -rf ${build_dir}/blaspp-pitzer-v100-build
CXX=$(which CC) cmake -S ${SRC_DIR}/blaspp \
  -B ${build_dir}/blaspp-pitzer-v100-build \
  -Duse_openmp=ON \
  -Dgpu_backend=cuda \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake --build ${build_dir}/blaspp-pitzer-v100-build --target install --parallel 16
rm -rf ${build_dir}/blaspp-pitzer-v100-build

# LAPACK++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/lapackpp ]; then
  cd ${SRC_DIR}/lapackpp
  git fetch
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git ${SRC_DIR}/lapackpp
fi
rm -rf ${build_dir}/lapackpp-pitzer-v100-build
CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S ${SRC_DIR}/lapackpp \
  -B ${build_dir}/lapackpp-pitzer-v100-build \
  -DCMAKE_CXX_STANDARD=17 \
  -Dbuild_tests=OFF \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake --build ${build_dir}/lapackpp-pitzer-v100-build --target install --parallel 16
rm -rf ${build_dir}/lapackpp-pitzer-v100-build

# c-blosc (I/O compression, for openPMD)
if [ -d ${SRC_DIR}/c-blosc ]; then
  cd ${SRC_DIR}/c-blosc
  git fetch --prune
  git checkout v1.21.6
  cd -
else
  git clone -b v1.21.6 https://github.com/Blosc/c-blosc.git ${SRC_DIR}/c-blosc
fi
rm -rf ${build_dir}/c-blosc-pitzer-build
cmake -S ${SRC_DIR}/c-blosc \
  -B ${build_dir}/c-blosc-pitzer-build \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DDEACTIVATE_AVX2=OFF \
  -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.6
cmake --build ${build_dir}/c-blosc-pitzer-build --target install --parallel 16
rm -rf ${build_dir}/c-blosc-pitzer-build

# ADIOS2 (for openPMD)
if [ -d ${SRC_DIR}/adios2 ]; then
  cd ${SRC_DIR}/adios2
  git fetch --prune
  git checkout v2.10.1
  cd -
else
  git clone -b v2.10.1 https://github.com/ornladios/ADIOS2.git ${SRC_DIR}/adios2
fi
rm -rf ${build_dir}/adios2-pitzer-build
cmake -S ${SRC_DIR}/adios2 \
  -B ${build_dir}/adios2-pitzer-build \
  -DBUILD_TESTING=OFF \
  -DADIOS2_BUILD_EXAMPLES=OFF \
  -DADIOS2_USE_Blosc=ON \
  -DADIOS2_USE_Fortran=OFF \
  -DADIOS2_USE_Python=OFF \
  -DADIOS2_USE_SST=OFF \
  -DADIOS2_USE_ZeroMQ=OFF \
  -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.10.1
cmake --build ${build_dir}/adios2-pitzer-build --target install -j 16
rm -rf ${build_dir}/adios2-pitzer-build

rm -rf ${build_dir}

# Python ######################################################################
#
python3 -m pip install --upgrade --user virtualenv
rm -rf ${SW_DIR}/venvs/${VENV_NAME}
python3 -m venv ${SW_DIR}/venvs/${VENV_NAME}
source ${SW_DIR}/venvs/${VENV_NAME}/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip cache purge
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

# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r ${SRC_DIR}/warpx/requirements.txt

# ML dependencies
python3 -m pip install --upgrade torch
