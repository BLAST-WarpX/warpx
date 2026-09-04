#!/usr/bin/env bash
#
# Configure and build WarpX through CTest, so that CDash records configure and
# build timings as well as compiler warnings and errors.
#
# Usage:
#   Tools/CI/ctest_dashboard.sh <build-dir> [cmake option ...]
#
# The CMake options are passed to the configure step unchanged; quote them as
# usual, values with spaces or semicolons are preserved.  Set the generator in
# CMAKE_GENERATOR rather than passing -G.  See Tools/CI/ctest_dashboard.cmake
# for the environment this expects.

set -o nounset -o errexit -o pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $(basename "$0") <build-dir> [cmake option ...]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$1"
CDASH_BUILD_DIR="$(cd -- "$1" && pwd)"
shift

# Hand the options over shell-quoted, for separate_arguments(UNIX_COMMAND) to
# parse back into exactly these arguments on the CMake side.
CDASH_CMAKE_ARGS=""
if [ $# -gt 0 ]; then
    CDASH_CMAKE_ARGS="$(printf '%q ' "$@")"
fi

export CDASH_BUILD_DIR CDASH_CMAKE_ARGS
export CDASH_BUILD_NAME="${CDASH_BUILD_NAME:?CDASH_BUILD_NAME must be set}"
export CDASH_SITE="${CDASH_SITE:?CDASH_SITE must be set}"
export CDASH_SUBMIT="${CDASH_SUBMIT:-OFF}"

exec ctest -VV -S "${script_dir}/ctest_dashboard.cmake"
