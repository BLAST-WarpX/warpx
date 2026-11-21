#!/usr/bin/env bash
#
# Copyright 2021 The WarpX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

# Parse GCC version from the command line (default: 12)
GCC_VERSION=${1:-12}

"$(dirname "$0")/ubuntu_apt.sh"

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
    ninja-build         \
    pkg-config          \
    wget

# ccache
$(dirname "$0")/ccache.sh
