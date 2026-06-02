#!/usr/bin/env python3

"""
Create two openPMD files for initializing WarpX particle momenta on a 3D Cartesian grid:

- ``example-u-std.h5`` with mesh ``u_std`` and components ``x``, ``y``, ``z``
- ``example-u-mean.h5`` with mesh ``u_mean`` and components ``x``, ``y``, ``z``

The component values are normalized momentum components, ``u = gamma * v / c``.
This test uses the same grid as ``inputs_test_3d_initial_distribution``.
"""

import numpy as np
import openpmd_api as io

x_1d = np.linspace(-1.0, 1.0, 8)
y_1d = np.linspace(-1.0, 1.0, 8)
z_1d = np.linspace(-1.0, 1.0, 8)
x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

grid_spacing = np.array(
    [
        x_1d[1] - x_1d[0],
        y_1d[1] - y_1d[0],
        z_1d[1] - z_1d[0],
    ]
)
grid_offset = [x_1d.min(), y_1d.min(), z_1d.min()]


def write_vector_mesh_file(filename, mesh_name, components):
    series = io.Series(filename, io.Access.create)
    it = series.iterations[1]

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
    del series


u_std_val = 3.1622776601683795e-05
write_vector_mesh_file(
    "new-example-u-std.h5",
    "u_std",
    {
        "x": np.full_like(x, u_std_val, dtype=np.float64),
        "y": np.full_like(y, u_std_val, dtype=np.float64),
        "z": np.full_like(z, u_std_val, dtype=np.float64),
    },
)

write_vector_mesh_file(
    "new-example-u-mean.h5",
    "u_mean",
    {
        "x": np.zeros_like(x, dtype=np.float64),
        "y": 0.2 * (z + 1.0) / 2.0,
        "z": np.zeros_like(z, dtype=np.float64),
    },
)
