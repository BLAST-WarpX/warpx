#!/usr/bin/env python3
"""
This script is used to test the results of the multi-J PSATD
first-order equations, with one J deposition. It compares the
energy of the electric field with a given reference energy.

The reference energy is computed by running the same test with J constant
in time, rho linear in time, and without divergence cleaning. The reference
energy corresponds to unstable results due to NCI (suppressed by the use of
both J and rho constant in time, and with divergence cleaning).
"""

import sys

import numpy as np
import scipy.constants as scc
import yt

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]

ds = yt.load(filename)

# yt 4.0+ has rounding issues with our domain data:
# RuntimeError: yt attempted to read outside the boundaries
# of a non-periodic domain along dimension 0.
if "force_periodicity" in dir(ds):
    ds.force_periodicity()

all_data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Ex = all_data["boxlib", "Ex"].squeeze().v
Ey = all_data["boxlib", "Ey"].squeeze().v
Ez = all_data["boxlib", "Ez"].squeeze().v

# Set reference energy values, and tolerances for numerical stability and charge conservation
tol_energy = 1e-8
energy_ref = 66e6

# Check numerical stability by comparing electric field energy to reference energy
energy = np.sum(scc.epsilon_0 / 2 * (Ex**2 + Ey**2 + Ez**2))
err_energy = energy / energy_ref
print("\nCheck numerical stability:")
print(f"err_energy = {err_energy}")
print(f"tol_energy = {tol_energy}")
assert err_energy < tol_energy
