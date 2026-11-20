#!/usr/bin/env python3

"""
Create an openPMD file containing the density with which
the WarpX particles should be initialized (on a 1D spherical (RSPHERE) grid)
"""

import numpy as np
import openpmd_api as io

on_axis_density = 1e24  # m^-3
channel_radius = 40e-6  # m

# Extend beyond the particle domain, providing neighbors for interpolation.
r = np.linspace(0.0, 50.0e-6, 50)
density_data = on_axis_density * (1 + r**2 / channel_radius**2)
series = io.Series("example-density-rsphere.h5", io.Access.create)

# only 1 iteration needed
it = series.iterations[1]
# set meta information
density = it.meshes["density"]

density.grid_spacing = [float(r[1] - r[0])]
density.grid_global_offset = [float(r[0])]
density.axis_labels = ["r"]
density.geometry = io.Geometry.spherical
density.unit_dimension = {io.Unit_Dimension.L: -3}

component = density[io.Mesh_Record_Component.SCALAR]
component.position = [0.0]
component.reset_dataset(io.Dataset(density_data.dtype, density_data.shape))
component.store_chunk(density_data)

series.flush()
series.close()
