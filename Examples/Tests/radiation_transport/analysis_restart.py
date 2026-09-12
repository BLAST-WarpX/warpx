#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
parser.add_argument("plotfile")
parser.add_argument("--compare-reference", action="store_true")
args = parser.parse_args()


def read_fields(plotfile):
    plotfile = Path(plotfile)
    with open(plotfile / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(
        str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names
    )


fields = read_fields(args.plotfile)
radiation_energy = np.loadtxt("diags/radiation_energy.txt")
if radiation_energy.ndim == 1:
    radiation_energy = radiation_energy[np.newaxis, :]
diffusion_fields = sorted(
    name for name in fields if name.startswith("radiation_diffusion_energy")
)
assert diffusion_fields
assert sum(np.sum(fields[name]) for name in diffusion_fields) > 0.0
assert np.any(fields["Te"] < np.max(fields["Te"]))
assert np.isfinite(radiation_energy[-1, 6])

if args.compare_reference:
    reference_dir = Path.cwd().with_name(Path.cwd().name.removesuffix("_restart"))
    reference = read_fields(reference_dir / args.plotfile)
    for name in ("rho", "Te", "radiation_material_energy", *diffusion_fields):
        print(f"comparing restart field {name}")
        np.testing.assert_allclose(fields[name], reference[name], rtol=2.0e-13)
    reference_radiation_energy = np.loadtxt(
        reference_dir / "diags/radiation_energy.txt"
    )
    print("comparing restart-safe radiation-energy diagnostic")
    np.testing.assert_allclose(
        radiation_energy[-1, 2:],
        reference_radiation_energy[-1, 2:],
        rtol=2.0e-13,
    )
