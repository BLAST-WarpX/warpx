#!/usr/bin/env python3
"""Validate cylindrical QCPML absorption and restart of convolution memories."""

import argparse
from pathlib import Path

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("--reference")
args = parser.parse_args()
yt.funcs.mylog.setLevel(50)
data = np.loadtxt("diags/reducedfiles/field_energy.txt")
# Initialization writes step zero before the fixture seeds the pulse.
energy = data[data[:, 0] > 0, 2::3]
assert np.isfinite(energy).all()
if not args.reference:
    for level in (0, 1):
        relative = energy[:, level] / energy[0, 0]
        print(f"level {level}: peak energy {relative.max()}, remaining {relative[-1]}")
        assert relative.max() < 1.1
        assert relative[-1] < 0.01
else:
    actual_dataset = yt.load("diags/diag1000350")
    reference_dataset = yt.load(str(Path(args.reference) / "diags/diag1000350"))
    actual = actual_dataset.all_data()
    reference = reference_dataset.all_data()
    for name in ("Er", "Et", "Ez", "Br", "Bt", "Bz"):
        key = next(key for key in actual.ds.field_list if key[1] == name)
        expected = np.asarray(reference[key])
        values = np.asarray(actual[key])
        scale = max(np.max(np.abs(expected)), 1.0e-30)
        error = np.max(np.abs(values - expected)) / scale
        print(f"restart relative {name} error: {error}")
        assert error < 2.0e-10
