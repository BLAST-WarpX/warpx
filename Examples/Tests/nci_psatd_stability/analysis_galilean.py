#!/usr/bin/env python3
"""
This script is used to test the results of the Galilean PSATD and
averaged Galilean PSATD methods in WarpX.

It compares the energy of the electric field with a given reference energy.

The reference energy is computed by running the same test with:
(i)  psatd.v_galilean=(0,0,0) for Galilean tests, or
(ii) psatd.do_time_averaging=0 for averaged Galilean tests.

In both cases, the reference energy corresponds to unstable results due to NCI
(suppressed by the Galilean PSATD method, without or with averaging, respectively).
"""

import re
import sys

import numpy as np
import scipy.constants as scc
import yt

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]

# Parse some input arguments from output file 'warpx_used_inputs'
current_correction = False
time_averaging = False
periodic_single_box = False
with open("./warpx_used_inputs", "r") as f:
    warpx_used_inputs = f.read()
if re.search("geometry.dims\s*=\s*2", warpx_used_inputs):
    dims = "2D"
elif re.search("geometry.dims\s*=\s*RZ", warpx_used_inputs):
    dims = "RZ"
elif re.search("geometry.dims\s*=\s*3", warpx_used_inputs):
    dims = "3D"
if re.search("psatd.current_correction\s*=\s*1", warpx_used_inputs):
    current_correction = True
if re.search("psatd.do_time_averaging\s*=\s*1", warpx_used_inputs):
    time_averaging = True
if re.search("psatd.periodic_single_box_fft\s*=\s*1", warpx_used_inputs):
    periodic_single_box = True

ds = yt.load(filename)

# yt 4.0+ has rounding issues with our domain data:
# RuntimeError: yt attempted to read outside the boundaries
# of a non-periodic domain along dimension 0.
if "force_periodicity" in dir(ds):
    ds.force_periodicity()

all_data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
if dims == "RZ":
    Ex = all_data["boxlib", "Er"].squeeze().v
    Ey = all_data["boxlib", "Et"].squeeze().v
else:
    Ex = all_data["boxlib", "Ex"].squeeze().v
    Ey = all_data["boxlib", "Ey"].squeeze().v
Ez = all_data["boxlib", "Ez"].squeeze().v

# Set reference energy values, and tolerances for numerical stability and charge conservation
tol_energy = 1e-8
tol_charge = 1e-9
if dims == "2D":
    if not current_correction:
        energy_ref = 35657.41657683263
    if current_correction and periodic_single_box:
        energy_ref = 35024.0275199999
    if current_correction and not periodic_single_box:
        energy_ref = 35675.25563324745
        tol_energy = 2e-8
        tol_charge = 2e-4
    if time_averaging:
        energy_ref = 26208.04843478073
        tol_energy = 1e-6
elif dims == "RZ":
    if not current_correction:
        energy_ref = 191002.6526271543
    if current_correction and periodic_single_box:
        energy_ref = 472779.70801323955
    if current_correction and not periodic_single_box:
        energy_ref = 511671.4108624746
        tol_charge = 3e-4
elif dims == "3D":
    if not current_correction:
        energy_ref = 661285.098907683
    if current_correction and periodic_single_box:
        energy_ref = 856783.3007547935
    if current_correction and not periodic_single_box:
        energy_ref = 875307.5138913819
        tol_charge = 1e-2
    if time_averaging:
        energy_ref = 14.564631643496
        tol_energy = 1e-4

# Check numerical stability by comparing electric field energy to reference energy
energy = np.sum(scc.epsilon_0 / 2 * (Ex**2 + Ey**2 + Ez**2))
err_energy = energy / energy_ref
print("\nCheck numerical stability:")
print(f"err_energy = {err_energy}")
print(f"tol_energy = {tol_energy}")
assert err_energy < tol_energy

# Check charge conservation (relative L-infinity norm of error) with current correction
if current_correction:
    divE = all_data["boxlib", "divE"].squeeze().v
    rho = all_data["boxlib", "rho"].squeeze().v / scc.epsilon_0
    err_charge = np.amax(np.abs(divE - rho)) / max(np.amax(divE), np.amax(rho))
    print("\nCheck charge conservation:")
    print(f"err_charge = {err_charge}")
    print(f"tol_charge = {tol_charge}")
    assert err_charge < tol_charge
