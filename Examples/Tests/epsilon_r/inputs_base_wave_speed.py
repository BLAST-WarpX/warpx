#!/usr/bin/env python3
"""Check reduced-wave-speed input, automatic CFL, and a vacuum standing wave."""

import argparse
import json
import sys

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument(
    "--parameter", choices=["epsilon_r", "c_light_factor", "default"], required=True
)
parser.add_argument("--factor", type=float, default=2)
args = parser.parse_args()
factor = 1 if args.parameter == "default" else args.factor
cells, cfl, steps = 32, 0.4, 8
c = 299792458.0
parameters = [
    "geometry.dims = 2",
    "geometry.prob_lo = 0 0",
    "geometry.prob_hi = 1 1",
    f"amr.n_cell = {cells} {cells}",
    f"amr.max_grid_size = {cells}",
    "amr.blocking_factor = 8",
    "amr.max_level = 0",
    "algo.maxwell_solver = Yee",
    "algo.particle_shape = 1",
    "boundary.field_lo = periodic periodic",
    "boundary.field_hi = periodic periodic",
    "boundary.particle_lo = periodic periodic",
    "boundary.particle_hi = periodic periodic",
    "particles.species_names = empty",
    "empty.species_type = electron",
    "empty.injection_style = none",
    "warpx.use_filter = 0",
    "warpx.verbose = 0",
    "warpx.do_dive_cleaning = 0",
    "warpx.do_divb_cleaning = 0",
    f"warpx.cfl = {cfl}",
    f"max_step = {steps}",
    "diagnostics.enable = 0",
    "warpx.B_ext_grid_init_style = parse_B_ext_grid_function",
    "warpx.Bx_external_grid_function(x,y,z) = 0",
    'warpx.By_external_grid_function(x,y,z) = "cos(2*pi*x)"',
    "warpx.Bz_external_grid_function(x,y,z) = 0",
]
if args.parameter != "default":
    value = factor**2 if args.parameter == "epsilon_r" else factor
    parameters.append(f"warpx.{args.parameter} = {value:.17g}")
geometry.dims = 2
libwarpx.initialize([sys.executable, *parameters])
try:
    dt = libwarpx.warpx.getdt(0)
    expected_dt = cfl * factor / (cells * c * np.sqrt(2))
    np.testing.assert_allclose(dt, expected_dt, rtol=2e-7, atol=0)
    register = libwarpx.warpx.multifab_register()
    initial_b = np.asarray(register.get("Bfield_fp", dir="y", level=0)[...]).copy()
    libwarpx.warpx.evolve(steps)
    expected_b = initial_b * np.cos(2 * np.pi * c / factor * steps * dt)
    observed_b = np.asarray(register.get("Bfield_fp", dir="y", level=0)[...])
    # Account for the second-order Yee dispersion at 32 cells per wavelength.
    np.testing.assert_allclose(observed_b, expected_b, rtol=0, atol=1e-3)
    snapshots = {
        f"{field}{direction}": np.asarray(
            register.get(f"{field}field_fp", dir=direction, level=0)[...]
        ).copy()
        for field in "EB"
        for direction in "xyz"
    }
    np.savez("fields.npz", **snapshots)
    with open("result.json", "w") as stream:
        json.dump(
            {
                "factor": factor,
                "dt": dt,
                "max_B_error": float(np.abs(observed_b - expected_b).max()),
            },
            stream,
            indent=2,
        )
finally:
    libwarpx.finalize()
