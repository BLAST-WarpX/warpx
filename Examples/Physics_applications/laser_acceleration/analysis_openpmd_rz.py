#!/usr/bin/env python3

import sys

import numpy as np
import openpmd_api as io

filename = sys.argv[1]
series = io.Series(f"{filename}/openpmd_%T.h5", io.Access.read_only)

assert len(series.iterations) == 3, "improper number of iterations stored"

ii = series.iterations[20]

assert len(ii.meshes) == 8, "improper number of meshes"

# select j_t
jt = ii.meshes["j"]["t"]

# this is in C (Python) order; r is the fastest varying index
(Nm, Nz, Nr) = jt.shape

assert (
    Nm == 3
), "Wrong number of angular modes stored or possible incorrect ordering when flushed"
assert (
    Nr == 64
), "Wrong number of radial points stored or possible incorrect ordering when flushed"
assert (
    Nz == 512
), "Wrong number of z points stored or possible incorrect ordering when flushed"

assert ii.meshes["part_per_grid"][io.Mesh_Record_Component.SCALAR].shape == [
    512,
    64,
], "problem with part_per_grid"
assert ii.meshes["rho_electrons"][io.Mesh_Record_Component.SCALAR].shape == [
    3,
    512,
    64,
], "problem with rho_electrons"


### test that openpmd+RZ
### 1. creates rho per species correctly
### 2. orders these appropriately
rhoe_mesh = ii.meshes["rho_electrons"]
rhob_mesh = ii.meshes["rho_beam"]
dz, dr = rhoe_mesh.grid_spacing
zmin, rmin = rhoe_mesh.grid_global_offset

rhoe = rhoe_mesh[io.Mesh_Record_Component.SCALAR][:]
rhob = rhob_mesh[io.Mesh_Record_Component.SCALAR][:]
series.flush()
nm, nz, nr = rhoe.shape
zlist = zmin + dz * np.arange(nz)
rhoe0 = rhoe[0]  # 0 mode
rhob0 = rhob[0]  # 0 mode

electron_meanz = np.sum(np.dot(zlist, rhoe0)) / np.sum(rhoe0)
beam_meanz = np.sum(np.dot(zlist, rhob0)) / np.sum(rhob0)

assert (
    (electron_meanz > 0) and (beam_meanz < 0)
), "problem with openPMD+RZ.  Maybe openPMD+RZ mixed up the order of rho_<species> diagnostics?"
