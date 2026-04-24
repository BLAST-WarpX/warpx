#!/usr/bin/env python3

import glob
import os
import sys

import numpy as np
import openpmd_api as io
import yaml


def load_field_from_iteration(
    series, iteration: int, field: str, coord: str = None
) -> np.ndarray:
    """Load iteration of field data from file."""

    it = series.iterations[iteration]
    field_obj = it.meshes[f"{field}"]

    if field_obj.scalar:
        field_data = field_obj[io.Mesh_Record_Component.SCALAR].load_chunk()
    elif coord in [item[0] for item in list(field_obj.items())]:
        field_data = field_obj[coord].load_chunk()
    else:
        raise Exception(
            f"Specified coordinate: f{coord} is not available for field: f{field}."
        )
    series.flush()

    return field_data


def compare_time_avg_with_instantaneous_diags(dir_inst: str, dir_avg: str):
    """Compare instantaneous data (multiple iterations averaged in post-processing) with in-situ averaged data."""

    field = "E"
    coord = "z"
    avg_period_steps = 5
    avg_output_step = 100

    path_tpl_inst = f"{dir_inst}/openpmd_%T.h5"
    path_tpl_avg = f"{dir_avg}/openpmd_%T.h5"

    si = io.Series(path_tpl_inst, io.Access.read_only)
    sa = io.Series(path_tpl_avg, io.Access.read_only)

    ii0 = si.iterations[0]
    fi0 = ii0.meshes[field][coord]
    shape = fi0.shape

    data_inst = np.zeros(shape)

    for i in np.arange(avg_output_step - avg_period_steps + 1, avg_output_step + 1):
        data_inst += load_field_from_iteration(si, i, field, coord)

    data_inst = data_inst / avg_period_steps

    data_avg = load_field_from_iteration(sa, avg_output_step, field, coord)

    # Compare the data
    if np.allclose(data_inst, data_avg, rtol=1e-12):
        print("Test passed: actual data is close to expected data.")
    else:
        print("Test failed: actual data is not close to expected data.")
        sys.exit(1)


def check_memory_per_rank_yaml(mpr_dir: str, prefix: str = "MPR"):
    """Smoke test for the MemoryPerRank YAML output.

    Validates that one file per rank exists, that every YAML document in each
    file parses cleanly to a plain dict, and that the key fields we document
    are present with the expected types. This catches YAML format regressions
    in the C++ emitter.
    """
    files = sorted(glob.glob(os.path.join(mpr_dir, f"{prefix}.*.yaml")))
    if not files:
        print(
            f"Test failed: no MemoryPerRank YAML files found in {mpr_dir} "
            f"(prefix={prefix})."
        )
        sys.exit(1)

    required_top_level = {"step", "time", "mpi", "host", "arenas"}

    for fpath in files:
        with open(fpath) as fh:
            docs = [d for d in yaml.safe_load_all(fh) if d is not None]
        if not docs:
            print(f"Test failed: {fpath} contains no YAML documents.")
            sys.exit(1)
        for d in docs:
            if not isinstance(d, dict):
                print(f"Test failed: {fpath} document is not a dict: {type(d)}.")
                sys.exit(1)
            missing = required_top_level - d.keys()
            if missing:
                print(
                    f"Test failed: {fpath} is missing required keys {missing}. "
                    f"Got keys: {sorted(d.keys())}"
                )
                sys.exit(1)
            if "main" not in (d.get("arenas") or {}):
                print(f"Test failed: {fpath} is missing arenas.main.")
                sys.exit(1)

    print(
        f"Test passed: MemoryPerRank YAML is valid across {len(files)} "
        f"rank files in {mpr_dir}."
    )


if __name__ == "__main__":
    # TODO: implement intervals parser for PICMI that allows more complex output periods
    test_name = os.path.split(os.getcwd())[1]
    if "picmi" not in test_name:
        # Functionality test for TimeAveragedDiagnostics
        compare_time_avg_with_instantaneous_diags(
            dir_inst=sys.argv[1],
            dir_avg="diags/diagTimeAvg/",
        )

    # Smoke test for MemoryPerRank YAML output (present in both variants).
    check_memory_per_rank_yaml("diags/reducedfiles/MemoryPerRank/", prefix="MPR")
