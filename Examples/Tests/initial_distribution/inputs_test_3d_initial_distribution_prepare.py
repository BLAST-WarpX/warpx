#!/usr/bin/env python3

"""
Create two openPMD files for initializing WarpX particle momenta on a 3D Cartesian grid:

- ``example-u-std.h5`` with fields ``ux_std``, ``uy_std``, ``uz_std``
- ``example-u-mean.h5`` with fields ``ux_mean``, ``uy_mean``, ``uz_mean``

This test uses the same grid as ``inputs_test_3d_initial_distribution``.
"""

import numpy as np
import openpmd_api as io

# -----------------------------
# Define the grid
# -----------------------------
x_1d = np.linspace(-1, 1, 8)
y_1d = np.linspace(-1, 1, 8)
z_1d = np.linspace(-1, 1, 8)
x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

dx = x_1d[1] - x_1d[0]
dy = y_1d[1] - y_1d[0]
dz = z_1d[1] - z_1d[0]

grid_spacing = np.array([dx, dy, dz])
grid_offset = [x_1d.min(), y_1d.min(), z_1d.min()]


# -----------------------------
# Helper function to write one file
# -----------------------------
def write_openpmd_file(filename, field_data):
    series = io.Series(filename, io.Access.create)
    it = series.iterations[1]

    for name, data in field_data.items():
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

    series.flush()
    del series


# -----------------------------
# Thermal spread
# -----------------------------
u_std_val = 3.1622776601683795e-05
std_fields = {
    "ux_std": np.ones_like(x) * u_std_val,
    "uy_std": np.ones_like(y) * u_std_val,
    "uz_std": np.ones_like(z) * u_std_val,
}

write_openpmd_file("example-u-std.h5", std_fields)


# -----------------------------
# Bulk velocity
# -----------------------------
mean_fields = {
    "ux_mean": np.zeros_like(x),
    "uy_mean": np.ones_like(y) * 0.2 * (z + 1) / 2,
    "uz_mean": np.zeros_like(z),
}

write_openpmd_file("example-u-mean.h5", mean_fields)
