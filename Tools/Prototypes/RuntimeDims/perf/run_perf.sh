#!/usr/bin/env bash
#
# Copyright 2026 Axel Huebl
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Performance comparison harness for the runtime-dimensionality prototype:
# runs warpx.unified vs. the native warpx.{1d,2d,3d} binaries on identical
# Langmuir problems and reports total times and TinyProfiler hot regions.
#
# Usage: run_perf.sh <build_dir> [steps] [threads]
set -eu

build_dir=${1:?usage: run_perf.sh <build_dir> [steps] [threads]}
steps=${2:-200}
threads=${3:-1}

repo=$(cd "$(dirname "$0")"/../../../.. && pwd)
[ -d "${repo}/Examples/Tests/langmuir" ] || repo=$(cd "$(dirname "$0")"/../../.. && pwd)
langmuir="${repo}/Examples/Tests/langmuir"
bin="${build_dir}/bin"
scratch=$(mktemp -d /tmp/warpx_perf.XXXXXX)
echo "# scratch: ${scratch}; steps: ${steps}; OMP threads: ${threads}"

overrides=(
    algo.current_deposition=direct
    diagnostics.diags_names=diag1
    diag1.intervals=1000000
    diag1.diag_type=Full
    "max_step=${steps}"
    warpx.verbose=0
)

run_one () {
    local label=$1 exe=$2 inputs=$3 dir=$4; shift 4
    mkdir -p "${dir}"
    (cd "${dir}" && cp "${langmuir}/$(basename "${inputs}")" .
     for b in "${langmuir}"/inputs_base_*; do cp "$b" . 2> /dev/null || true; done
     OMP_NUM_THREADS=${threads} "${exe}" "$(basename "${inputs}")" "${overrides[@]}" "$@" \
         > run.log 2>&1)
    local total
    total=$(grep -m1 -oE "TinyProfiler total time across processes.*" "${dir}/run.log" | grep -oE "[0-9.]+" | head -1)
    echo "${label}: total ${total} s"
    # top exclusive TinyProfiler regions
    grep -m1 -A8 "Excl. Min" "${dir}/run.log" | tail -7 | sed "s/^/    /"
}

# problem sizes per dimensionality: <dim> <n_cell override> <grid override> <ppc>
cases=(
    "1 amr.n_cell=1048576 amr.max_grid_size=1048576 8"
    "2 amr.n_cell=1024,1024 amr.max_grid_size=1024 4"
    "3 amr.n_cell=128,128,128 amr.max_grid_size=128 2"
)

for c in "${cases[@]}"; do
    set -- $c
    D=$1; ncell=${2//,/ }; mgs=$3; ppc=$4
    inputs="${langmuir}/inputs_test_${D}d_langmuir_multi"
    ppc_arg=""
    case ${D} in
        1) ppc_arg="electrons.num_particles_per_cell_each_dim=${ppc} positrons.num_particles_per_cell_each_dim=${ppc}";;
        2) ppc_arg="electrons.num_particles_per_cell_each_dim=${ppc} ${ppc} positrons.num_particles_per_cell_each_dim=${ppc} ${ppc}";;
        3) ppc_arg="electrons.num_particles_per_cell_each_dim=${ppc} ${ppc} ${ppc} positrons.num_particles_per_cell_each_dim=${ppc} ${ppc} ${ppc}";;
    esac
    echo
    echo "## ${D}D: ${ncell}, ppc=${ppc}, ${steps} steps"
    # shellcheck disable=SC2086
    run_one "unified ${D}d" "${bin}/warpx.unified" "${inputs}" "${scratch}/u${D}" \
        "${ncell}" "${mgs}" ${ppc_arg}
    # shellcheck disable=SC2086
    run_one "native  ${D}d" "${bin}/warpx.${D}d" "${inputs}" "${scratch}/n${D}" \
        "${ncell}" "${mgs}" ${ppc_arg}
done
