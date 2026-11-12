#!/usr/bin/env bash
#
# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

set -euo pipefail

echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

# GitHub and Azure Ubuntu runners can temporarily lose access to their regional
# mirror. Probe once before apt retries the same host for every archive.
mirrors=${1:-/etc/apt/apt-mirrors.txt}
[[ -r "$mirrors" ]] || exit 0
regional=$(awk '$1 ~ /^https?:\/\/azure[.]archive[.]ubuntu[.]com\/ubuntu\/?$/ {print $1; exit}' "$mirrors")
archive=$(awk '$1 ~ /^https:\/\/archive[.]ubuntu[.]com\/ubuntu\/?$/ {print $1; exit}' "$mirrors")
[[ -n "$regional" && -n "$archive" ]] || exit 0

source /etc/os-release
serves_release () {
    curl --fail --silent --show-error --location --head --connect-timeout 3 --max-time 8 \
        "${1%/}/dists/${VERSION_CODENAME}/InRelease" >/dev/null
}

if ! serves_release "$regional" && serves_release "$archive"; then
    echo "Using the runner's Ubuntu archive mirror because $regional is unavailable."
    sudo sed -i -E '/^[[:space:]]*https?:\/\/azure[.]archive[.]ubuntu[.]com\/ubuntu\/?([[:space:]]|$)/d' "$mirrors"
fi
