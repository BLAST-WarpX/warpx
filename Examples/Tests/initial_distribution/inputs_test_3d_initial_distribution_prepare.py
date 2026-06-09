#!/usr/bin/env python3

"""
Create openPMD files containing the mean and standard deviation of the
normalized momentum components with which WarpX particles should be initialized
(on a Cartesian grid).
"""

import numpy as np
import openpmd_api as io

# Define u_mean and u_std as functions of x, y, z, using numpy syntax
# - Define the grid
x_1d = np.linspace(-1.0, 1.0, 8)
y_1d = np.linspace(-1.0, 1.0, 8)
z_1d = np.linspace(-1.0, 1.0, 8)
x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

# - Define the normalized momentum data, u = gamma * v / c
ux_std_data = 0.2 * abs(z)
uy_std_data = 0.21 * abs(z)
uz_std_data = 0.22 * abs(z)


ux_mean_data = np.zeros_like(x, dtype=np.float64)
uy_mean_data = 0.2 * (z + 1.0) / 2.0
uz_mean_data = np.zeros_like(x, dtype=np.float64)

grid_spacing = np.array(
    [
        x_1d[1] - x_1d[0],
        y_1d[1] - y_1d[0],
        z_1d[1] - z_1d[0],
    ]
)
grid_offset = [x_1d.min(), y_1d.min(), z_1d.min()]


def write_vector_mesh_file(filename, mesh_name, components):
    # create openPMD file
    series = io.Series(filename, io.Access.create)
    # only 1 iteration needed
    it = series.iterations[1]

    # set meta information
    mesh = it.meshes[mesh_name]
    mesh.grid_spacing = grid_spacing
    mesh.grid_global_offset = grid_offset
    mesh.axis_labels = ["x", "y", "z"]
    mesh.geometry = io.Geometry.cartesian
    mesh.unit_dimension = {}

    for component_name, data in components.items():
        component = mesh[component_name]
        component.position = [0.0, 0.0, 0.0]
        component.reset_dataset(io.Dataset(data.dtype, data.shape))
        component.store_chunk(data)

    series.flush()


write_vector_mesh_file(
    "example-u-std.h5",
    "u_std",
    {
        "x": ux_std_data,
        "y": uy_std_data,
        "z": uz_std_data,
    },
)

write_vector_mesh_file(
    "example-u-mean.h5",
    "u_mean",
    {
        "x": ux_mean_data,
        "y": uy_mean_data,
        "z": uz_mean_data,
    },
)
