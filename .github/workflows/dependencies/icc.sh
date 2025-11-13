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

export DEBIAN_FRONTEND=noninteractive
sudo apt-get -qqq update
sudo apt-get install -y \
  build-essential \
  ca-certificates \
  cmake           \
  gnupg           \
  pkg-config      \
  wget

# ccache
$(dirname "$0")/ccache.sh

# Ref.: https://github.com/rscohn2/oneapi-ci
sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | \
    sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update

# try apt install up to five times, to avoid connection splits
status=1
for itry in {1..5}
do
    sudo apt-get install -y                              \
        intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic  \
        && { sudo apt-get clean; status=0; break; }  \
        || { sleep 10; }
done
if [[ ${status} -ne 0 ]]; then exit 1; fi

# activate now via
set +eu
source /opt/intel/oneapi/setvars.sh
set -eu

# cmake-easyinstall
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
sudo chmod a+x /usr/local/bin/cmake-easyinstall
export CEI_SUDO="sudo"
export CEI_TMP="/tmp/cei"

# openPMD-api
CXX=$(which icpc) CC=$(which icc) \
  cmake-easyinstall               \
  --prefix=/usr/local             \
  git+https://github.com/openPMD/openPMD-api.git@0.16.1 \
  -DopenPMD_USE_PYTHON=OFF \
  -DBUILD_TESTING=OFF      \
  -DBUILD_EXAMPLES=OFF     \
  -DBUILD_CLI_TOOLS=OFF    \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$(which ccache)
