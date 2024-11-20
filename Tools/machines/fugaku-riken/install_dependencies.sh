#!/bin/bash
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl, Luca Fedeli
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail

# Remove old dependencies #####################################################
#
SRC_DIR="${HOME}/src/"
SW_DIR="${HOME}/sw/fugaku/a64fx/"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}
mkdir -p ${SRC_DIR}

# General extra dependencies ##################################################
#

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
  rm -rf ${SRC_DIR}/c-blosc-fugaku-build
  cmake -S ${SRC_DIR}/c-blosc -B ${SRC_DIR}/c-blosc-fugaku-build -DBUILD_SHARED_LIBS=OFF -DBUILD_SHARED=OFF -DBUILD_STATIC=ON -DBUILD_TESTS=OFF -DBUILD_FUZZERS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1-install
  cmake --build ${SRC_DIR}/c-blosc-fugaku-build --target install --parallel 48
  rm -rf ${SRC_DIR}/c-blosc-fugaku-build

# ADIOS2 (I/O)
if [ -d ${SRC_DIR}/c-blosc ]
then
  cd ${SRC_DIR}/adios2
  git fetch --prune
  git checkout v2.8.3
  cd -
else
  git clone -b v2.8.3 https://github.com/ornladios/ADIOS2.git ${SRC_DIR}/adios2
fi
rm -rf ${SRC_DIR}/adios2-fugaku-build
cmake -S ${SRC_DIR}/adios2 -B ${SRC_DIR}/adios2-fugaku-build -DBUILD_SHARED_LIBS=OFF -DADIOS2_USE_Blosc=ON -DBUILD_TESTING=OFF -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3-install
cmake --build ${SRC_DIR}/adios2-fugaku-build --target install -j 48
rm -rf ${SRC_DIR}/adios2-fugaku-build
