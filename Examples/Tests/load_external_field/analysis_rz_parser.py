#!/usr/bin/env python3

# Copyright 2024 WarpX
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This test checks that the external B and E field parser works in RZ geometry.
# The input file initializes the m=0 (axisymmetric) external fields using the
# parser, with the (x, y, z) coordinates mapped to (r, 0, z):
#   Br = 0.5*z,  Bt = 0.1,  Bz = 1.0
#   Er = 2.0*r,  Et = 0.0,  Ez = -0.5*z
# This script reads the field data from the diagnostic output and compares it
# against the analytical expressions.

# Possible error: 0.0
# tolerance: 1.0e-12
# Possible running time: <1 s

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

tolerance = 1.0e-12

filename = sys.argv[1]
ds = yt.load(filename)

# The grid is collocated, so all fields are cell centered.
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# Cell-center coordinates in (r, z). dims = (nr, nz, 1) for RZ in yt.
nr, nz, _ = ds.domain_dimensions
dr = (ds.domain_right_edge[0] - ds.domain_left_edge[0]).d / nr
dz = (ds.domain_right_edge[1] - ds.domain_left_edge[1]).d / nz
r = (np.arange(nr) + 0.5) * dr + ds.domain_left_edge[0].d
z = (np.arange(nz) + 0.5) * dz + ds.domain_left_edge[1].d
R, Z = np.meshgrid(r, z, indexing="ij")

# Analytical expressions (parser maps x->r, y->0, z->z).
Br_th = 0.5 * Z
Bt_th = np.full_like(Br_th, 0.1)
Bz_th = np.full_like(Br_th, 1.0)
Er_th = 2.0 * R
Et_th = np.full_like(Br_th, 0.0)
Ez_th = -0.5 * Z


def rel_error(sim, th):
    sim = sim.squeeze()
    # Slice to exclude boundary cells where cell-centered staggered grid values are halved
    sim_inner = sim[1:-1, 1:-1]
    th_inner = th[1:-1, 1:-1]
    denom = np.sqrt(np.sum(np.square(th_inner)))
    if denom == 0.0:
        return np.max(np.abs(sim_inner))
    return np.sqrt(np.sum(np.square(sim_inner - th_inner)) / denom)


Br_sim = data[("boxlib", "Br")].to_ndarray()
Bt_sim = data[("boxlib", "Bt")].to_ndarray()
Bz_sim = data[("boxlib", "Bz")].to_ndarray()
Er_sim = data[("boxlib", "Er")].to_ndarray()
Et_sim = data[("boxlib", "Et")].to_ndarray()
Ez_sim = data[("boxlib", "Ez")].to_ndarray()

errors = {
    "Br": rel_error(Br_sim, Br_th),
    "Bt": rel_error(Bt_sim, Bt_th),
    "Bz": rel_error(Bz_sim, Bz_th),
    "Er": rel_error(Er_sim, Er_th),
    "Et": rel_error(Et_sim, Et_th),
    "Ez": rel_error(Ez_sim, Ez_th),
}

max_error = max(errors.values())
print("errors:", {k: f"{v:.3e}" for k, v in errors.items()})
print("max error =", max_error)
print("tolerance =", tolerance)
assert max_error < tolerance
