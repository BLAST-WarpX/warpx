#!/usr/bin/env bash
#
# Configure and build WarpX through CTest, so that CDash records configure and
# build timings as well as compiler warnings and errors.
#
# Usage:
#   Tools/CI/ctest_dashboard.sh <build-dir> [cmake option ...]
#
# The CMake options are passed to the configure step unchanged; quote them as
# usual, values with spaces or semicolons are preserved.
#
# Environment:
#   CDASH_BUILD_NAME  dashboard build name, e.g. "CPU-3D" (required)
#   CDASH_SITE        dashboard site name, e.g. "Azure" (required)
#   CDASH_SUBMIT      "ON" submits to CDash right away; use this for build-only
#                     jobs.  Jobs that also run tests should leave it "OFF" and
#                     submit once at the end, after their
#                     `ctest ... -D ExperimentalTest` step, with
#                     `ctest --test-dir <build-dir> -D ExperimentalSubmit`.
#
# The build parallelism comes from CMAKE_BUILD_PARALLEL_LEVEL: CTest builds via
# `cmake --build`, which honors it. There is no `-j` for the CTest build step.

set -o nounset -o errexit -o pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $(basename "$0") <build-dir> [cmake option ...]" >&2
    exit 2
fi

build_dir="$1"
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd -- "${script_dir}/../.." && pwd)"

mkdir -p "${build_dir}"
CDASH_BINARY_DIR="$(cd -- "${build_dir}" && pwd)"
export CDASH_BINARY_DIR

export CDASH_SOURCE_DIR="${source_dir}"
export CDASH_BUILD_NAME="${CDASH_BUILD_NAME:?CDASH_BUILD_NAME must be set}"
export CDASH_SITE="${CDASH_SITE:?CDASH_SITE must be set}"
export CDASH_SUBMIT="${CDASH_SUBMIT:-OFF}"

# hand the options over in a file: one per line, no quoting games needed
CDASH_OPTIONS_FILE="${CDASH_BINARY_DIR}/.cdash_configure_options"
export CDASH_OPTIONS_FILE
: > "${CDASH_OPTIONS_FILE}"
if [ $# -gt 0 ]; then
    printf '%s\n' "$@" > "${CDASH_OPTIONS_FILE}"
fi
# BUILDNAME and SITE are what include(CTest) writes into DartConfiguration.tcl,
# from where a later `ctest -D ExperimentalTest` picks them up for Test.xml.
# Without them that step would label Test.xml with the host name and a generic
# build name, and CDash would file it as a build separate from Configure.xml and
# Build.xml. Appended last so they always match CTEST_BUILD_NAME/CTEST_SITE.
printf '%s\n' "-DBUILDNAME=${CDASH_BUILD_NAME}" "-DSITE=${CDASH_SITE}" \
    >> "${CDASH_OPTIONS_FILE}"

exec ctest -VV -S "${script_dir}/ctest_dashboard.cmake"
