#!/usr/bin/env bash

set -eu -o pipefail

if [[ $# -eq 2 ]]; then
  CVER=$1
else
  CVER=4.8.3
fi

wget https://github.com/ccache/ccache/releases/download/v${CVER}/ccache-${CVER}-linux-x86_64.tar.xz
tar xvf ccache-${CVER}-linux-x86_64.tar.xz
sudo cp -f ccache-${CVER}-linux-x86_64/ccache /usr/local/bin/
