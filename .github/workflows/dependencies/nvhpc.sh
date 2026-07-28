#!/usr/bin/env bash
#
# Copyright 2021-2022 The WarpX Community
#
# Author: Axel Huebl
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo apt -qqq update
sudo apt install -y     \
    build-essential     \
    ca-certificates     \
    cmake               \
    environment-modules \
    gnupg               \
    libhiredis-dev      \
    libzstd-dev         \
    ninja-build         \
    pkg-config          \
    wget

# ccache
$(dirname "$0")/ccache.sh

# parse version number from command line argument
VERSION_DOTTED=${1:-25.1}
VERSION_DASHED=${VERSION_DOTTED/./-}  # replace first occurence of "." with "-"

# install nvhpc
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
sudo apt update -y
sudo apt install -y --no-install-recommends nvhpc-${VERSION_DASHED}

# clean up space
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf /opt/nvidia/hpc_sdk/Linux_x86_64/${VERSION_DOTTED}/examples
sudo rm -rf /opt/nvidia/hpc_sdk/Linux_x86_64/${VERSION_DOTTED}/profilers
sudo rm -rf /opt/nvidia/hpc_sdk/Linux_x86_64/${VERSION_DOTTED}/math_libs/12.6/targets/x86_64-linux/lib/lib*_static*.a

# things should reside in /opt/nvidia/hpc_sdk now

# activation via:
#   source /etc/profile.d/modules.sh
#   module load /opt/nvidia/hpc_sdk/modulefiles/nvhpc/${VERSION_DOTTED}

# cmake-easyinstall
#
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
sudo chmod a+x /usr/local/bin/cmake-easyinstall
export CEI_SUDO="sudo"
export CEI_TMP="/tmp/cei"
