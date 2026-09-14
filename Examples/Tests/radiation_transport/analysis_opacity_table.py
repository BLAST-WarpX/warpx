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
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=10.0)
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

plotfile = Path("diags/diag1000001")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
cross_rtol = 5.0e-6 if cross_dtype == np.float32 else 3.0e-13
particle_energy = np.loadtxt("diags/particle_energy.txt")
radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

alpha = args.alpha
dt = 1.0e-10
initial_temperature = elementary_charge / Boltzmann
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
temperature_lo = 5.80225906077504e3
temperature_hi = 2.320903624310016e4
temperature_fraction = np.log(initial_temperature / temperature_lo) / np.log(
    temperature_hi / temperature_lo
)
alpha_planck = np.exp(np.log(10.0) + temperature_fraction * np.log(40.0 / 10.0))

initial_streaming = particle_energy[0, 4]
final_streaming = particle_energy[-1, 4]
expected_streaming = initial_streaming * np.exp(-alpha * c * dt)
expected_diffusion = (
    radiation_constant * initial_temperature**4 * (1.0 - np.exp(-alpha_planck * c * dt))
)

diffusion = fields["radiation_diffusion_energy"].squeeze()
diffusion_total = np.sum(diffusion)
material_exchange = np.sum(fields["radiation_material_energy"])
expected_material_exchange = initial_streaming - expected_streaming - expected_diffusion

print(
    f"streaming: simulated={final_streaming:.16e}, expected={expected_streaming:.16e}"
)
print(
    f"diffusion: simulated={diffusion_total:.16e}, expected={expected_diffusion:.16e}"
)
print(f"material exchange: {material_exchange:.16e} J")

np.testing.assert_allclose(final_streaming, expected_streaming, rtol=cross_rtol)
np.testing.assert_allclose(diffusion_total, expected_diffusion, rtol=cross_rtol)
np.testing.assert_allclose(
    material_exchange, expected_material_exchange, rtol=cross_rtol
)
np.testing.assert_allclose(
    final_streaming + diffusion_total + material_exchange,
    initial_streaming,
    rtol=cross_rtol,
)
np.testing.assert_allclose(
    diffusion, expected_diffusion / diffusion.size, rtol=cross_rtol
)
np.testing.assert_allclose(
    radiation_energy[-1, 2], final_streaming + diffusion_total, rtol=cross_rtol
)
np.testing.assert_allclose(radiation_energy[-1, 5], material_exchange, rtol=cross_rtol)
np.testing.assert_allclose(radiation_energy[-1, 6], material_exchange, rtol=cross_rtol)
np.testing.assert_allclose(radiation_energy[-1, 7:9], 0.0, atol=1.0e-12)
np.testing.assert_allclose(radiation_energy[-1, 9], material_exchange, rtol=cross_rtol)
np.testing.assert_allclose(radiation_energy[-1, 10], 0.0, atol=1.0e-12)
np.testing.assert_allclose(radiation_energy[-1, 11:15], 0.0, atol=1.0e-12)

roundoff_scale = max(
    np.max(np.abs(radiation_energy[:, 2:])),
    abs(material_exchange),
    np.finfo(np.float64).tiny,
)
ledger_atol = max(1.0e-24, 64.0 * np.finfo(cross_dtype).eps * roundoff_scale)
current_closure_residual = (
    radiation_energy[0, 2]
    - radiation_energy[-1, 2]
    - radiation_energy[-1, 5]
    - radiation_energy[-1, 7]
)
cumulative_closure_residual = (
    radiation_energy[0, 2]
    - radiation_energy[-1, 2]
    - radiation_energy[-1, 6]
    - radiation_energy[-1, 8]
)
np.testing.assert_allclose(
    radiation_energy[-1, 15], current_closure_residual, rtol=0.0, atol=ledger_atol
)
np.testing.assert_allclose(
    radiation_energy[-1, 16],
    cumulative_closure_residual,
    rtol=0.0,
    atol=ledger_atol,
)
np.testing.assert_allclose(
    radiation_energy[-1, 2]
    + radiation_energy[-1, 6]
    + radiation_energy[-1, 8]
    + radiation_energy[-1, 16],
    radiation_energy[0, 2],
    rtol=0.0,
    atol=ledger_atol,
)
