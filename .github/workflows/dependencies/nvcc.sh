#!/usr/bin/env bash
#
# Copyright 2020-2023 The WarpX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    gnupg               \
    libhiredis-dev      \
    libopenmpi-dev      \
    libzstd-dev         \
    ninja-build         \
    openmpi-bin         \
    pkg-config          \
    wget

# ccache
$(dirname "$0")/ccache.sh

# parse version number from command line argument
VERSION_DOTTED=${1-12.6} && VERSION_DASHED=$(sed 's/\./-/' <<< ${VERSION_DOTTED})

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get install -y          \
    cuda-command-line-tools-${VERSION_DASHED} \
    cuda-compiler-${VERSION_DASHED}           \
    cuda-cupti-dev-${VERSION_DASHED}          \
    cuda-minimal-build-${VERSION_DASHED}      \
    cuda-nvml-dev-${VERSION_DASHED}           \
    cuda-nvtx-${VERSION_DASHED}               \
    libcufft-dev-${VERSION_DASHED}            \
    libcurand-dev-${VERSION_DASHED}           \
    libcusparse-dev-${VERSION_DASHED}
sudo ln -s cuda-${VERSION_DOTTED} /usr/local/cuda

# if we run out of temporary storage in CI:
#du -sh /usr/local/cuda-${VERSION_DOTTED}
#echo "+++ REDUCING CUDA Toolkit install size +++"
#sudo rm -rf /usr/local/cuda-${VERSION_DOTTED}/targets/x86_64-linux/lib/libcu{fft,pti,rand}_static.a
#sudo rm -rf /usr/local/cuda-${VERSION_DOTTED}/targets/x86_64-linux/lib/libnvperf_host_static.a
#du -sh /usr/local/cuda-${VERSION_DOTTED}/
#df -h

# cmake-easyinstall
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
sudo chmod a+x /usr/local/bin/cmake-easyinstall
export CEI_SUDO="sudo"
export CEI_TMP="/tmp/cei"
