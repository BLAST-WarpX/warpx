#!/usr/bin/env python3

"""
Create an openPMD file containing the temperature profile (in eV) used
to initialize particles on a 1D spherical mesh.
"""

import numpy as np
import openpmd_api as io
from scipy.constants import c as clight
from scipy.constants import electron_mass as m_e
from scipy.constants import elementary_charge as q_e

r_1d = np.linspace(0.0, 1.0, 16)
u_std = 0.2 * (0.5 + r_1d)

temperature_in_eV = u_std**2 * m_e * clight**2 / q_e

series = io.Series("example-temperature-in-eV.h5", io.Access.create)
mesh = series.iterations[1].meshes["temperature_in_eV"]
mesh.grid_spacing = [float(r_1d[1] - r_1d[0])]
mesh.grid_global_offset = [float(r_1d[0])]
mesh.axis_labels = ["r"]
mesh.geometry = io.Geometry.spherical
mesh.unit_dimension = {}

component = mesh[io.Mesh_Record_Component.SCALAR]
component.position = [0.0]
component.reset_dataset(io.Dataset(temperature_in_eV.dtype, temperature_in_eV.shape))
component.store_chunk(temperature_in_eV)
series.flush()
series.close()
