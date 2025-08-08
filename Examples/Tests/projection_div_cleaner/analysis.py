#!/usr/bin/env python3

# Copyright 2022 Yinjian Zhao
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This test tests the external field loading feature.
# A magnetic mirror field is loaded, and a single particle
# in the mirror will be reflected by the magnetic mirror effect.
# At the end of the simulation, the position of the particle
# is compared with known correct results.

# Possible errors: 6.235230443866285e-9
# tolerance: 1.0e-8
# Possible running time: 0.327827743 s

import sys

import numpy as np
import yt

tolerance = 4e-3

filename = sys.argv[1]

ds = yt.load(filename)
grid0 = ds.index.grids[0]


r_min = grid0.LeftEdge.v[0]
r_max = grid0.RightEdge.v[0]
nr = grid0.shape[0]
dri = 1 // grid0.dds.v[0]
ir = np.arange(nr)

ru = 1.0 + 0.5 / (r_min * dri + ir + 0.5)
rd = 1.0 - 0.5 / (r_min * dri + ir + 0.5)

z_min = grid0.LeftEdge.v[1]
z_max = grid0.RightEdge.v[1]
nz = grid0.shape[1]
dzi = 1.0 / grid0.dds.v[1]

RU, ZU = np.meshgrid(ru, np.arange(nz), indexing="ij")
RD, ZD = np.meshgrid(rd, np.arange(nz), indexing="ij")

dBrdr = (
    RU * grid0["raw", "Bx_aux"].v[:, :, 0, 1]
    - RD * grid0["raw", "Bx_aux"].v[:, :, 0, 0]
) * dri
dBzdz = (
    grid0["raw", "Bz_aux"].v[:, :, 0, 1] - grid0["raw", "Bz_aux"].v[:, :, 0, 0]
) * dzi

divB = dBrdr + dBzdz

import matplotlib.pyplot as plt

plt.imshow(np.log10(np.abs(divB[1:-1, 1:-1])))
plt.title("log10(|div(B)|)")
plt.colorbar()
plt.savefig("divb.png")

error = np.sqrt((divB[1:-1, 1:-1] ** 2).sum())

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
