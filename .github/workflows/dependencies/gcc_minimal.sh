#!/usr/bin/env bash
#
# Copyright 2023 The WarpX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

"$(dirname "$0")/ubuntu_apt.sh"

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    gnupg               \
    ninja-build         \
    pkg-config          \
    wget

# ccache
$(dirname "$0")/ccache.sh
