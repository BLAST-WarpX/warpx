#!/usr/bin/env python3
"""
Create an openPMD file containing the density with which
the WarpX particles should be initialized (on a 3D Cartesian grid)
"""

import numpy as np
import openpmd_api as io

density_data = np.zeros((41, 41, 41))
density_data[:, :, :] = np.arange(41)[np.newaxis, np.newaxis, :]

# create openpmd file
series = io.Series("example-density.h5", io.Access.create)
# only 1 iteratiion needed
it = series.iterations[1]
# set meta information
density = it.meshes["density"]
density.grid_spacing = np.array([1e-6, 1e-6, 1e-6])
density.grid_global_offset = [-20e-6, -20e-6, -20e-6]
density.axis_labels = ["x", "y", "z"]
density.geometry = io.Geometry.cartesian
density.unit_dimension = {
    io.Unit_Dimension.L: -3,
    io.Unit_Dimension.I: 1,
    io.Unit_Dimension.T: 1,
}

# label
density_d = density[io.Mesh_Record_Component.SCALAR]
density_d.position = [0, 0, 0]

dataset = io.Dataset(density_data.dtype, density_data.shape)
density_d.reset_dataset(dataset)
density_d.store_chunk(density_data)
series.flush()

del series
