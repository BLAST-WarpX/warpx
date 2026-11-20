#!/usr/bin/env python3
"""Reject ambiguous or invalid reduced-wave-speed input before advancing fields."""

import argparse
import sys

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case", choices=["both", "small_factor", "small_epsilon", "overflow"]
)
args = parser.parse_args()
cases = {
    "both": (
        ["warpx.c_light_factor = 2", "warpx.epsilon_r = 4"],
        "Specify only one of warpx.epsilon_r and warpx.c_light_factor",
    ),
    "small_factor": (
        ["warpx.c_light_factor = 0.5"],
        "warpx.c_light_factor must be finite, >= 1, and have a representable square",
    ),
    "small_epsilon": (
        ["warpx.epsilon_r = 0.5"],
        "warpx.epsilon_r must be finite and >= 1.0",
    ),
    "overflow": (
        ["warpx.c_light_factor = 1e200"],
        "warpx.c_light_factor must be finite, >= 1, and have a representable square",
    ),
}
extra, expected_message = cases[args.case]
parameters = [
    "amrex.throw_exception = 1",
    "geometry.dims = 2",
    "geometry.prob_lo = -1 -1",
    "geometry.prob_hi = 1 1",
    "amr.n_cell = 8 8",
    "amr.max_level = 0",
    "algo.maxwell_solver = Yee",
    "algo.particle_shape = 1",
    "boundary.field_lo = periodic periodic",
    "boundary.field_hi = periodic periodic",
    "particles.species_names = empty",
    "empty.species_type = electron",
    "empty.injection_style = none",
    "max_step = 0",
    "diagnostics.enable = 0",
    *extra,
]
geometry.dims = 2
try:
    libwarpx.initialize([sys.executable, *parameters])
except RuntimeError as error:
    # WarpX wraps messages and prefixes continuation lines with '#'.
    message = " ".join(str(error).replace("#", " ").split())
    assert expected_message in message, str(error)
    print(f"Correctly rejected {args.case}: {expected_message}")
else:
    raise AssertionError(f"Invalid {args.case} input was accepted")
finally:
    if libwarpx.initialized:
        libwarpx.finalize()
    else:
        # A constructor exception already destroyed the partial WarpX object.
        # Finalize only AMReX (including MPI), then end this test process.
        libwarpx.libwarpx_so.amrex_finalize()
