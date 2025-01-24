#!/usr/bin/env python3

# Copyright 2019-2020 Luca Fedeli, Maxence Thevenet, Revathi Jambunathan
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


"""
Analysis script of a WarpX simulation of rigid injection in a boosted frame.

A Gaussian electron beam starts from -5 microns, propagates rigidly up to
20 microns after which it expands due to emittance only (the focal position is
20 microns). The beam width is measured after ~50 microns, and compared with
the theory (with a 1% relative error allowed).

The simulation runs in a boosted frame, and the analysis is done in the lab
frame, i.e., on the back-transformed diagnostics.
"""

import sys

import numpy as np
import openpmd_api as io
import yt

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]

# Tolerances to check consistency between plotfile BTD and openPMD BTD
rtol = 1e-16
atol = 1e-16

# Read data from new back-transformed diagnostics (plotfile)
ds_plotfile = yt.load(filename)
x_plotfile = ds_plotfile.all_data()["beam", "particle_position_x"].v
z_plotfile = ds_plotfile.all_data()["beam", "particle_position_y"].v
ux_plotfile = ds_plotfile.all_data()["beam", "particle_momentum_x"].v
uy_plotfile = ds_plotfile.all_data()["beam", "particle_momentum_y"].v
uz_plotfile = ds_plotfile.all_data()["beam", "particle_momentum_z"].v

# Read data from new back-transformed diagnostics (openPMD)
series = io.Series("./diags/diag2/openpmd_%T.h5", io.Access.read_only)
ds_openpmd = series.iterations[1]
x_openpmd = ds_openpmd.particles["beam"]["position"]["x"][:]
z_openpmd = ds_openpmd.particles["beam"]["position"]["z"][:]
ux_openpmd = ds_openpmd.particles["beam"]["momentum"]["x"][:]
uy_openpmd = ds_openpmd.particles["beam"]["momentum"]["y"][:]
uz_openpmd = ds_openpmd.particles["beam"]["momentum"]["z"][:]
series.flush()

# Sort and compare arrays to check consistency between plotfile BTD and openPMD BTD
assert np.allclose(np.sort(x_plotfile), np.sort(x_openpmd), rtol=rtol, atol=atol)
assert np.allclose(np.sort(z_plotfile), np.sort(z_openpmd), rtol=rtol, atol=atol)
assert np.allclose(np.sort(ux_plotfile), np.sort(ux_openpmd), rtol=rtol, atol=atol)
assert np.allclose(np.sort(uy_plotfile), np.sort(uy_openpmd), rtol=rtol, atol=atol)
assert np.allclose(np.sort(uz_plotfile), np.sort(uz_openpmd), rtol=rtol, atol=atol)

# Initial parameters
z0 = 20.0e-6
x0 = 1.0e-6
theta0 = np.arcsin(0.1)

# Theoretical beam width after propagation with rigid injection
z = np.mean(z_plotfile)
x = np.std(x_plotfile)
print(f"Beam position = {z}")
print(f"Beam width    = {x}")

xth = np.sqrt(x0**2 + (z - z0) ** 2 * theta0**2)
err = np.abs((x - xth) / xth)
tol = 1e-2
print(f"error = {err}")
print(f"tolerance = {tol}")
assert err < tol
