#!/usr/bin/env python3
"""Check prescribed nodal material temperature and restart preservation."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, elementary_charge

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
field_dtype, _, _ = precision_dtypes(args)
rtol = 8.0e-6 if field_dtype == np.float32 else 3.0e-12


def load(path):
    with (path / "Header").open() as header:
        header.readline()
        names = [header.readline().strip() for _ in range(int(header.readline()))]
    return _read_buffer(str(path), str(path / "Level_0/Cell_H"), names)


if not args.restart:
    initial = load(Path("diags/diag1000000"))
    nodes = np.arange(17) / 16
    nodal_temperature = (
        (1 + 0.1 * np.sin(2 * np.pi * nodes)) * elementary_charge / Boltzmann
    )
    expected = 0.5 * (nodal_temperature[:-1] + nodal_temperature[1:])
    np.testing.assert_allclose(np.squeeze(initial["Te"]), expected, rtol=rtol)
    # Initial pressure must already represent the prescribed temperature,
    # rather than becoming nonzero only in the first PIC bootstrap.
    np.testing.assert_allclose(
        initial["Pe"],
        initial["rho"] / elementary_charge * Boltzmann * initial["Te"],
        rtol=rtol,
    )
    # Frozen ions, zero curl(B), and zero material exchange leave V_e = 0.
    # Repeated marker projection must not diffuse this stationary profile.
    final = load(Path("diags/diag1000002"))
    np.testing.assert_allclose(final["Te"], initial["Te"], rtol=rtol)
else:
    reference = Path(
        "../test_1d_radiation_transport_material_profile/diags/diag1000002"
    )
    expected = load(reference)
    actual = load(Path("diags/diag1000002"))
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=rtol, atol=0)
print("Material temperature initialization/restart passed")
