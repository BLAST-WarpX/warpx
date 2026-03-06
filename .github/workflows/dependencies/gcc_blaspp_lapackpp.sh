#!/usr/bin/env bash
#
# Copyright 2023 The WarpX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl, Luca Fedeli

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

# Parse GCC version from the command line (default: 12)
GCC_VERSION=${1:-12}

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    g++-${GCC_VERSION}  \
    gnupg               \
    libboost-math-dev   \
    libfftw3-dev        \
    libfftw3-mpi-dev    \
    libhdf5-openmpi-dev \
    libopenmpi-dev      \
    libblas-dev         \
    liblapack-dev       \
    ninja-build         \
    pkg-config          \
    wget

# ccache
$(dirname "$0")/ccache.sh

# cmake-easyinstall
#
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
sudo chmod a+x /usr/local/bin/cmake-easyinstall
export CEI_SUDO="sudo"
export CEI_TMP="/tmp/cei"

# BLAS++ & LAPACK++
cmake-easyinstall \
  --prefix=/usr/local                      \
  git+https://github.com/icl-utk-edu/blaspp.git \
  -Duse_openmp=OFF                        \
  -Dbuild_tests=OFF                        \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$(which ccache) \
  -DCMAKE_VERBOSE_MAKEFILE=ON

cmake-easyinstall \
  --prefix=/usr/local                        \
  git+https://github.com/icl-utk-edu/lapackpp.git \
  -Duse_cmake_find_lapack=ON                 \
  -Dbuild_tests=OFF                          \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$(which ccache) \
  -DCMAKE_VERBOSE_MAKEFILE=ON
