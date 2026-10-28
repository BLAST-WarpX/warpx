#!/usr/bin/env python3
"""Check conductive relaxation and driven Ohm's law against their analytic ODE."""

import argparse
import json
import sys

import numpy as np

from pywarpx import callbacks, geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["backwardeuler", "laxwendroff"], required=True)
parser.add_argument(
    "--parameter", choices=["epsilon_r", "c_light_factor", "default"], required=True
)
parser.add_argument("--factor", type=float, default=2)
parser.add_argument("--steps", type=int, default=256)
args = parser.parse_args()

epsilon0 = 8.8541878128e-12
epsilon = 2 * epsilon0
sigma = epsilon / 1e-10
relative_epsilon = 1 if args.parameter == "default" else args.factor**2
relaxation_time = relative_epsilon * epsilon / sigma
dt = relaxation_time / args.steps
initial_e = np.array([1.0, -0.5, 2.0])
equilibrium_e = np.array([0.0, 0.75, -1.25])
imposed_current = -sigma * equilibrium_e
parameters = [
    "geometry.dims = 2",
    "geometry.prob_lo = -1 -1",
    "geometry.prob_hi = 1 1",
    "amr.n_cell = 8 8",
    "amr.max_grid_size = 8",
    "amr.blocking_factor = 8",
    "amr.max_level = 0",
    "algo.maxwell_solver = Yee",
    "algo.em_solver_medium = macroscopic",
    f"algo.macroscopic_sigma_method = {args.method}",
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
    f"warpx.const_dt = {dt:.17g}",
    f"max_step = {args.steps}",
    "diagnostics.enable = 0",
    f"macroscopic.epsilon_function(x,y,z) = {epsilon:.17g}",
    f"macroscopic.sigma_function(x,y,z) = {sigma:.17g}",
    "macroscopic.mu_function(x,y,z) = mu0",
    "warpx.E_ext_grid_init_style = constant",
    "warpx.E_external_grid = " + " ".join(map(str, initial_e)),
]
if args.parameter != "default":
    value = relative_epsilon if args.parameter == "epsilon_r" else args.factor
    parameters.append(f"warpx.{args.parameter} = {value:.17g}")

geometry.dims = 2
libwarpx.initialize([sys.executable, *parameters])
register = libwarpx.warpx.multifab_register()


def supply_current():
    # There are no particles: this is a spatially constant impressed current.
    for direction, value in zip("xyz", imposed_current):
        register.get("current_fp", dir=direction, level=0).set_val(float(value))


callbacks.installbeforeEsolve(supply_current)
try:
    libwarpx.warpx.evolve(args.steps)
    expected = equilibrium_e + (initial_e - equilibrium_e) * np.exp(-1.0)
    # The ODE is epsilon_eff*dE/dt = -sigma*E - J. Bounds account for the
    # known first-/second-order temporal error, independently of solver alpha/beta.
    amplitude = float(np.max(np.abs(initial_e - equilibrium_e)))
    tolerance = (
        amplitude / args.steps
        if args.method == "backwardeuler"
        else amplitude / args.steps**2
    )
    snapshots = {}
    for i, direction in enumerate("xyz"):
        electric = np.asarray(register.get("Efield_fp", dir=direction, level=0)[...])
        magnetic = np.asarray(register.get("Bfield_fp", dir=direction, level=0)[...])
        snapshots[f"E{direction}"] = electric.copy()
        snapshots[f"B{direction}"] = magnetic.copy()
        np.testing.assert_allclose(electric, expected[i], rtol=0, atol=tolerance)
        np.testing.assert_allclose(magnetic, 0, rtol=0, atol=1e-12)
    np.savez("fields.npz", **snapshots)
    result = {
        "method": args.method,
        "parameter": args.parameter,
        "epsilon_r": relative_epsilon,
        "physical_sigma": sigma,
        "time_s": relaxation_time,
        "expected_E": expected.tolist(),
        "observed_E": [float(snapshots[f"E{d}"].mean()) for d in "xyz"],
        "analytic_tolerance": tolerance,
    }
    with open("result.json", "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
finally:
    libwarpx.finalize()
