#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action="store_true")
parser.add_argument("--multigroup", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
assert not (args.restart and args.multigroup)
field_dtype, _, _ = precision_dtypes(args)


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


num_cells = 16
cell_width = 1.0 / num_cells
z = (np.arange(num_cells) + 0.5) * cell_width
rtol = 8.0e-6 if field_dtype == np.float32 else 3.0e-13


def check_fields(fields: dict[str, np.ndarray], step: int) -> None:
    if args.multigroup:
        radiation_energy = np.squeeze(fields["radiation_diffusion_energy_g0"])
        empty_group = np.squeeze(fields["radiation_diffusion_energy_g1"])
        high_group = np.squeeze(fields["radiation_diffusion_energy_g2"])
        np.testing.assert_array_equal(empty_group, np.zeros_like(empty_group))
        np.testing.assert_allclose(high_group, (3.0 + 4.0 * z) * cell_width, rtol=rtol)
    else:
        radiation_energy = np.squeeze(fields["radiation_diffusion_energy"])

    np.testing.assert_allclose(
        radiation_energy, (1.0 + 2.0 * z) * cell_width, rtol=rtol
    )
    np.testing.assert_allclose(
        fields["radiation_material_energy"], 0.0, rtol=0.0, atol=0.0
    )
    assert np.all(np.isfinite(radiation_energy))

    print(f"step {step}: radiation={radiation_energy.sum():.16e} J")


steps = (0,) if args.multigroup else ((1, 2) if args.restart else (0, 1, 2))
fields_by_step = {step: load_plotfile(Path(f"diags/diag1{step:06d}")) for step in steps}
for step, fields in fields_by_step.items():
    check_fields(fields, step)

if args.restart:
    reference_dir = Path.cwd().with_name(Path.cwd().name.removesuffix("_restart"))
    for step, fields in fields_by_step.items():
        reference = load_plotfile(reference_dir / f"diags/diag1{step:06d}")
        names = ["radiation_material_energy", "radiation_diffusion_energy"]
        for name in names:
            np.testing.assert_array_equal(fields[name], reference[name])
