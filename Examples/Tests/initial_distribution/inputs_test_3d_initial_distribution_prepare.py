#!/usr/bin/env python3

"""
Create an openPMD file containing ux_mean, uy_mean, and uz_mean
used to initialize WarpX particle momentum on a 3D Cartesian grid.
"""

import numpy as np
import openpmd_api as io

# Define u_mean as a function of x, y, z, using numpy syntax
# - Define the grid
x_1d = np.linspace(-1, 1, 8)
y_1d = np.linspace(-1, 1, 8)
z_1d = np.linspace(-1, 1, 8)
x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
# - Define bulk velocity as a function of x, y, z
ux_data = np.zeros_like(x)
uy_data = np.ones_like(x) * 0.2 * (z + 1) / 2
uz_data = np.zeros_like(x)

# create openpmd file
series = io.Series("example-u-mean.h5", io.Access.create)
# only 1 iteration needed
it = series.iterations[1]

dx = x_1d[1] - x_1d[0]
dy = y_1d[1] - y_1d[0]
dz = z_1d[1] - z_1d[0]

grid_spacing = np.array([dx, dy, dz])
grid_offset = [x_1d.min(), y_1d.min(), z_1d.min()]


# -----------------------------
# Helper function to write mesh
# -----------------------------
def write_mesh(name, data):
    mesh = it.meshes[name]
    mesh.grid_spacing = grid_spacing
    mesh.grid_global_offset = grid_offset
    mesh.axis_labels = ["x", "y", "z"]
    mesh.geometry = io.Geometry.cartesian

    mesh.unit_dimension = {}
    comp = mesh[io.Mesh_Record_Component.SCALAR]
    comp.position = [0, 0, 0]

    dataset = io.Dataset(data.dtype, data.shape)
    comp.reset_dataset(dataset)
    comp.store_chunk(data)


# -----------------------------
# Write fields
# -----------------------------
write_mesh("ux_mean", ux_data)
write_mesh("uy_mean", uy_data)
write_mesh("uz_mean", uz_data)
series.flush()
del series
