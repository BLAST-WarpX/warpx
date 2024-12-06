#!/usr/bin/env bash
#
# Copyright 2024 The WarpX Community
#
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

# This dependency file is currently used within a docker container,
# which does not come (among others) with wget, xz-utils, curl, git,
# ccache, and pkg-config pre-installed.
sudo apt-get -qq update
sudo apt-get install -y \
    cmake               \
    libblas-dev         \
    libc++-17-dev       \
    libboost-math-dev   \
    libfftw3-dev        \
    libfftw3-mpi-dev    \
    libhdf5-openmpi-dev \
    liblapack-dev       \
    libopenmpi-dev      \
    libomp-17-dev       \
    ninja-build         \
    wget                \
    xz-utils            \
    curl                \
    git                 \
    ccache              \
    pkg-config

# parse clang version from command line
version_number=${1}
# add LLVM repository and install clang tools
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh ${version_number}
sudo apt-get update
sudo apt-get install clang-${version_number} clang-tidy-${version_number}
# export compiler flags
export CXX=$(which clang++-${version_number})
export CC=$(which clang-${version_number})

# cmake-easyinstall
#
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
sudo chmod a+x /usr/local/bin/cmake-easyinstall
export CEI_SUDO="sudo"
export CEI_TMP="/tmp/cei"

# BLAS++ & LAPACK++
cmake-easyinstall \
  --prefix=/usr/local                           \
  git+https://github.com/icl-utk-edu/blaspp.git \
  -Duse_openmp=OFF                              \
  -Dbuild_tests=OFF                             \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$(which ccache) \
  -DCMAKE_VERBOSE_MAKEFILE=ON

cmake-easyinstall \
  --prefix=/usr/local                             \
  git+https://github.com/icl-utk-edu/lapackpp.git \
  -Duse_cmake_find_lapack=ON                      \
  -Dbuild_tests=OFF                               \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$(which ccache)   \
  -DCMAKE_VERBOSE_MAKEFILE=ON
