#!/usr/bin/env python3

# Copyright 2021 Roelof Groenewald

# This script tests the time dependent Dirichlet boundary
# conditions in a 2D electrostatic simulation. An empty
# domain of 64 x 8 cells is simulated with periodic boundary
# conditions in the x directions and Dirichlet boundary
# conditions in the y direction with specified potentials
# of sine waves with different periods on the lo and hi side.
# One period of the hi side sine wave is simulated and the
# potentials at the boundaries compared to expectation.

# Possible running time: ~ 19 s

import glob

import numpy as np
import yt

files = sorted(glob.glob("diags/diag1*"))[1:]
assert len(files) > 0

times = np.ones(len(files))
potentials_lo = np.zeros(len(files))
potentials_hi = np.zeros(len(files))

for ii, file in enumerate(files):
    ds = yt.load(file)
    times[ii] = ds.current_time.item()
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    potentials_lo[ii] = np.mean(data["phi"].to_ndarray()[0])
    potentials_hi[ii] = np.mean(data["phi"].to_ndarray()[-1])

expected_potentials_lo = 150.0 * np.sin(2.0 * np.pi * 6.78e6 * times)
expected_potentials_hi = 450.0 * np.sin(2.0 * np.pi * 13.56e6 * times)

assert np.allclose(potentials_lo, expected_potentials_lo, rtol=0.1)
assert np.allclose(potentials_hi, expected_potentials_hi, rtol=0.1)
