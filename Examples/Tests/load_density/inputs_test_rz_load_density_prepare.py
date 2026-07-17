#!/usr/bin/env python3
"""
Create an openPMD file containing the density with which
the WarpX particles should be initialized (on a 2D axisymmetric (RZ) grid)
"""

import numpy as np
import openpmd_api as io

# Define density as a function of r, z, using numpy syntax
# parabolic channel in r, with a ramp and plateau in z
on_axis_density = 1e24  # m^-3
channel_radius = 40e-6  # m
ramp_length = 60e-6  # m
# - Define the grid
num_nodes = 1
nr = 200
nz = 200
r_1d = np.linspace(0, 50e-6, nr)
z_1d = np.linspace(0, 500e-6, nz)
r, z = np.meshgrid(r_1d, z_1d, indexing="ij")
# - Define density as a function of r, z
density_data = (
    on_axis_density
    * (1 + r**2 / channel_radius**2)
    * np.where(z < ramp_length, z / ramp_length, 1)
).reshape(num_nodes, nr, nz)

# create openpmd file
series = io.Series("example-density.h5", io.Access.create)
# only 1 iteration needed
it = series.iterations[1]
# set meta information
density = it.meshes["density"]
density.grid_spacing = np.array([r_1d[1] - r_1d[0], z_1d[1] - z_1d[0]])
density.grid_global_offset = [r_1d.min(), z_1d.min()]
density.axis_labels = ["r", "z"]
density.geometry = io.Geometry.thetaMode
density.unit_dimension = {
    io.Unit_Dimension.L: -3,
}

# label
density_d = density[io.Mesh_Record_Component.SCALAR]
density_d.position = [0, 0, 0]

dataset = io.Dataset(density_data.dtype, density_data.shape)
density_d.reset_dataset(dataset)
density_d.store_chunk(density_data)
series.flush()

del series
