#!/usr/bin/env python3
"""
Create an openPMD file containing the density with which
the WarpX particles should be initialized (on a 1D Cartesian grid)
"""

import numpy as np
import openpmd_api as io

# Define density as a function of z using numpy syntax
# with a ramp and plateau in z
on_axis_density = 1e24  # m^-3
ramp_length = 60e-6  # m
# - Define the grid
z = np.linspace(0, 500e-6, 100)
# - Define density as a function of z
density_data = np.where(
    z < ramp_length, z * on_axis_density / ramp_length, on_axis_density
)
# create openpmd file
series = io.Series("example-density.h5", io.Access.create)

# only 1 iteration needed
it = series.iterations[1]
# set meta information
density = it.meshes["density"]
density.grid_spacing = np.array([z[1] - z[0]])
density.grid_global_offset = [z.min()]
density.axis_labels = ["z"]
density.geometry = io.Geometry.cartesian
density.unit_dimension = {
    io.Unit_Dimension.L: -3,
}

# label
density_d = density[io.Mesh_Record_Component.SCALAR]
density_d.position = [0]

dataset = io.Dataset(density_data.dtype, density_data.shape)
density_d.reset_dataset(dataset)
density_d.store_chunk(density_data)
series.flush()

series.close()
