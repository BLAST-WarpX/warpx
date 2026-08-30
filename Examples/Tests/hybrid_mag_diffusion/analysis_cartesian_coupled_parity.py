#!/usr/bin/env python3
"""Compare a coupled Cartesian PETSc solve with its AMReX GMRES reference."""

import glob
import math
import os
import re
import sys

PARITY_TOL = 2.0e-6


def reference_dir():
    test_name = os.path.basename(os.getcwd())
    if not test_name.endswith("_petsc"):
        raise RuntimeError(f"PETSc test directory expected, got {test_name}")
    return os.path.join("..", test_name.removesuffix("_petsc"))


def last_plotfile(base_dir):
    plots = sorted(
        path
        for path in glob.glob(os.path.join(base_dir, "diags", "diag1*"))
        if os.path.isdir(path)
    )
    if not plots:
        raise RuntimeError(f"No plotfiles under {base_dir}/diags/")
    return plots[-1]


def component_minmax(diag_dir):
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    with open(cell_h) as infile:
        text = infile.read()
    float_lines = []
    for line in text.splitlines():
        if not re.match(r"^-?[0-9]", line.strip()):
            continue
        try:
            values = [
                float(value) for value in line.strip().rstrip(",").split(",") if value
            ]
        except ValueError:
            continue
        if len(values) >= 3:
            float_lines.append(values)
    if len(float_lines) < 2 or len(float_lines) % 2 != 0:
        raise RuntimeError(f"Could not parse field min/max from {cell_h}")
    num_boxes = len(float_lines) // 2
    box_mins = float_lines[:num_boxes]
    box_maxs = float_lines[num_boxes:]
    global_min = [min(row[component] for row in box_mins) for component in range(3)]
    global_max = [max(row[component] for row in box_maxs) for component in range(3)]
    return global_min, global_max


def magnetic_energy(base_dir):
    path = os.path.join(base_dir, "diags", "reduced_files", "field_energy.txt")
    with open(path) as infile:
        rows = [
            line
            for line in infile
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not rows:
        raise RuntimeError(f"No data rows in {path}")
    return float(rows[-1].split()[4])


def relative_error(a, b, scale):
    return abs(a - b) / max(scale, 1.0e-30)


def main():
    amrex_ref_dir = reference_dir()
    ref_plot = last_plotfile(amrex_ref_dir)
    petsc_plot = last_plotfile(".")
    ref_min, ref_max = component_minmax(ref_plot)
    petsc_min, petsc_max = component_minmax(petsc_plot)
    ref_energy = magnetic_energy(amrex_ref_dir)
    petsc_energy = magnetic_energy(".")

    values = ref_min + ref_max + petsc_min + petsc_max + [ref_energy, petsc_energy]
    if not all(math.isfinite(value) for value in values):
        print("FAILED: non-finite field extrema or magnetic energy")
        return 1

    names = ["By", "Bx", "Bz"]
    errors = []
    field_scale = max(abs(value) for value in ref_min + ref_max + petsc_min + petsc_max)
    for component, name in enumerate(names):
        scale = max(
            abs(ref_min[component]),
            abs(ref_max[component]),
            abs(petsc_min[component]),
            abs(petsc_max[component]),
            field_scale * 1.0e-10,
        )
        error = max(
            relative_error(ref_min[component], petsc_min[component], scale),
            relative_error(ref_max[component], petsc_max[component], scale),
        )
        errors.append(error)
        print(
            f"{name}: AMReX=[{ref_min[component]:.9e}, {ref_max[component]:.9e}] "
            f"PETSc=[{petsc_min[component]:.9e}, {petsc_max[component]:.9e}] "
            f"rel={error:.3e}"
        )

    energy_error = relative_error(
        ref_energy, petsc_energy, max(abs(ref_energy), abs(petsc_energy))
    )
    print(
        f"B-energy: AMReX={ref_energy:.12e} PETSc={petsc_energy:.12e} "
        f"rel={energy_error:.3e}"
    )

    if max(errors + [energy_error]) >= PARITY_TOL:
        print(f"FAILED: PETSc/AMReX parity error >= {PARITY_TOL:.0e}")
        return 1

    print(
        "PASSED: Cartesian coupled PETSc/AMReX parity and exact-Pmat "
        f"field agreement < {PARITY_TOL:.0e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
