#!/usr/bin/env bash
#
# Copyright 2020-2021 The WarpX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

# Ref.: https://github.com/rscohn2/oneapi-ci
# intel-basekit intel-hpckit are too large in size

# download the key to system keyring
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# add signed entry to apt sources and configure the APT client to use Intel repository:
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt-get update

df -h
# Install and reduce disk space
# https://github.com/BLAST-WarpX/warpx/pull/1566#issuecomment-790934878

# try apt install up to five times, to avoid connection splits
# FIXME install latest version of IntelLLVM, Intel MKL
#       after conflicts with openPMD are resolved
status=1
for itry in {1..5}
do
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        cmake           \
        intel-oneapi-compiler-dpcpp-cpp=2024.2.1-1079 \
        intel-oneapi-mkl-devel=2024.2.1-103 \
        g++ gfortran    \
        libopenmpi-dev  \
        openmpi-bin     \
        && { sudo apt-get clean; status=0; break; }  \
        || { sleep 10; }
done
if [[ ${status} -ne 0 ]]; then exit 1; fi

du -sh /opt/intel/oneapi/
du -sh /opt/intel/oneapi/*/*
echo "+++ REDUCING oneAPI install size +++"
sudo rm -rf /opt/intel/oneapi/mkl/latest/lib/intel64/*.a           \
            /opt/intel/oneapi/compiler/latest/linux/lib/oclfpga    \
            /opt/intel/oneapi/compiler/latest/linux/lib/emu        \
            /opt/intel/oneapi/compiler/latest/linux/bin/intel64    \
            /opt/intel/oneapi/compiler/latest/linux/bin/lld        \
            /opt/intel/oneapi/compiler/latest/linux/bin/lld-link   \
            /opt/intel/oneapi/compiler/latest/linux/bin/wasm-ld
du -sh /opt/intel/oneapi/
du -sh /opt/intel/oneapi/*/*
df -h

# ccache
$(dirname "$0")/ccache.sh
