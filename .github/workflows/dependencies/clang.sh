#!/usr/bin/env bash
#
# Copyright 2025 The WarpX Community
#
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

# `man apt.conf`: number of retries to perform (if non-zero,
# APT will retry failed files the given number of times).
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo apt-get update
sudo apt-get install -y \
    cmake               \
    libblas-dev         \
    libboost-math-dev   \
    libfftw3-dev        \
    libfftw3-mpi-dev    \
    libhdf5-openmpi-dev \
    liblapack-dev       \
    libopenmpi-dev      \
    ninja-build

# parse clang version number from command line
version_number=${1}

# add LLVM repository and install clang tools
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh ${version_number}

# install clang, clang-tidy, and
# LLVM implementations of C++ standard library and OpenMP
sudo apt-get update
sudo apt-get install -y          \
    clang-${version_number}      \
    clang-tidy-${version_number} \
    libc++-${version_number}-dev \
    libomp-${version_number}-dev

# export compiler flags
export CXX=$(which clang++-${version_number})
export CC=$(which clang-${version_number})

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
