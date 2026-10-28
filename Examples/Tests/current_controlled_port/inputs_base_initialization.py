#!/usr/bin/env python3
"""Check the native port circulation before and after the first field push."""

import argparse
import sys
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx, picmi

parser = argparse.ArgumentParser()
parser.add_argument("--amplitude", type=float, required=True)
parser.add_argument("--current", type=float, required=True)
args = parser.parse_args()

# A later nonzero point satisfies the waveform API's nonzero-drive requirement
# while leaving the target constant throughout this one-step control.
np.savetxt(
    "initial_current_profile.txt",
    [[0, args.current], [1e-8, args.current], [2e-8, args.current + 1]],
)
parameters = [
    "geometry.dims = 3",
    "geometry.prob_lo = -0.5 -0.5 -0.5",
    "geometry.prob_hi = 0.5 0.5 0.5",
    "amr.n_cell = 16 16 16",
    "amr.max_grid_size = 16",
    "amr.blocking_factor = 8",
    "amr.max_level = 0",
    "algo.maxwell_solver = Yee",
    "algo.evolve_scheme = explicit",
    "algo.particle_shape = 1",
    "algo.current_deposition = esirkepov",
    "boundary.field_lo = periodic periodic periodic",
    "boundary.field_hi = periodic periodic periodic",
    "boundary.particle_lo = periodic periodic periodic",
    "boundary.particle_hi = periodic periodic periodic",
    "particles.species_names = empty",
    "empty.species_type = electron",
    "empty.injection_style = none",
    "warpx.do_dive_cleaning = 0",
    "warpx.do_divb_cleaning = 0",
    "warpx.use_filter = 0",
    "warpx.verbose = 0",
    "warpx.const_dt = 1e-12",
    "max_step = 1",
    "diagnostics.enable = 0",
    "warpx.B_ext_grid_init_style = parse_B_ext_grid_function",
    'warpx.Bx_external_grid_function(x,y,z) = "0"',
    'warpx.By_external_grid_function(x,y,z) = "0"',
    f'warpx.Bz_external_grid_function(x,y,z) = "{args.amplitude}*sin(2*pi*y)"',
    "warpx.current_controlled_port = 1",
    "warpx.current_controlled_port.file = initial_current_profile.txt",
    "warpx.current_controlled_port.direction = 0",
    "warpx.current_controlled_port.terminal_0.lower_bound = -0.21875 -0.25 -0.25",
    "warpx.current_controlled_port.terminal_0.upper_bound = -0.21875 0.25 0.25",
    "warpx.current_controlled_port.terminal_1.lower_bound = 0.21875 -0.25 -0.25",
    "warpx.current_controlled_port.terminal_1.upper_bound = 0.21875 0.25 0.25",
]
Path("initial_inputs").write_text("\n".join(parameters) + "\n")
geometry.dims = 3
# CTest prefixes input-file lookups with the source directory. Pass the generated
# parameters directly while retaining the deck above in this test's run directory.
libwarpx.initialize([sys.executable, *parameters])


def check_current(label):
    register = libwarpx.warpx.multifab_register()
    by = np.asarray(register.get("Bfield_fp", dir="y", level=0)[...]).squeeze()
    bz = np.asarray(register.get("Bfield_fp", dir="z", level=0)[...]).squeeze()
    dx = 1 / 16
    mu0 = picmi.constants.mu0
    # The Ex surface includes transverse nodes 4..12; its bounding staggered
    # B edges have transverse indices 3 and 12. Measure both terminals directly
    # rather than trusting the port's cached status.
    measured = np.array(
        [
            dx
            / mu0
            * (
                by[i, 4:13, 3].sum()
                + bz[i, 12, 4:13].sum()
                - by[i, 4:13, 12].sum()
                - bz[i, 3, 4:13].sum()
            )
            for i in (4, 11)
        ]
    )
    external_current = (
        args.amplitude
        * 9
        * dx
        / mu0
        * (np.sin(2 * np.pi * 0.28125) - np.sin(-2 * np.pi * 0.28125))
    )
    tolerance = 2e-12 * max(1.0, abs(args.current), abs(external_current))
    status = np.asarray(libwarpx.warpx.get_current_controlled_port_status())
    print(f"{label}: native currents={measured}, status={status}", flush=True)
    assert np.max(np.abs(measured - args.current)) < tolerance
    assert np.max(np.abs(status - args.current)) < tolerance


try:
    # This assertion must run before evolve: the first normal push repairs the
    # old startup inconsistency and would otherwise hide the regression.
    check_current("initial")
    libwarpx.warpx.evolve(1)
    check_current("after one step")
finally:
    libwarpx.finalize()
