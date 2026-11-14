#!/usr/bin/env python3

"""
Create an openPMD file containing the temperature (in eV) with which
WarpX particles should be initialized, in RZ, RCYLINDER or RSPHERE geometry.

The temperature is bilinear in (r, z) (RZ) or linear in r (RCYLINDER, RSPHERE),
so that the linear interpolation done by WarpX is exact, and the analysis
script can compare with the analytical expression `temperature_in_eV`.
"""

import argparse

import numpy as np
import openpmd_api as io

T0 = 100.0  # eV


def temperature_in_eV(r, z=0.0):
    """Analytical temperature profile, as a function of r (and z in RZ)"""
    return T0 * (1.0 + 2.0 * r) * (1.5 + 0.5 * z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry", choices=["rz", "rcylinder", "rsphere"], required=True
    )
    args = parser.parse_args()

    r_1d = np.linspace(0.0, 1.0, 11)
    dr = r_1d[1] - r_1d[0]

    if args.geometry == "rz":
        z_1d = np.linspace(-1.0, 1.0, 11)
        r, z = np.meshgrid(r_1d, z_1d, indexing="ij")
        # The first axis is the azimuthal mode (only mode 0 here)
        data = temperature_in_eV(r, z).reshape(1, r_1d.size, z_1d.size)
        geometry = io.Geometry.thetaMode
        axis_labels = ["r", "z"]
        grid_spacing = [dr, z_1d[1] - z_1d[0]]
        grid_offset = [r_1d.min(), z_1d.min()]
        position = [0.0, 0.0]
    else:
        data = temperature_in_eV(r_1d)
        if args.geometry == "rcylinder":
            geometry = io.Geometry.cylindrical
        else:
            geometry = io.Geometry.spherical
        axis_labels = ["r"]
        grid_spacing = [dr]
        grid_offset = [r_1d.min()]
        position = [0.0]

    series = io.Series("example-temperature-in-eV.h5", io.Access.create)
    # only 1 iteration needed
    it = series.iterations[1]

    mesh = it.meshes["temperature_in_eV"]
    mesh.grid_spacing = grid_spacing
    mesh.grid_global_offset = grid_offset
    mesh.axis_labels = axis_labels
    mesh.geometry = geometry
    mesh.unit_dimension = {}

    component = mesh[io.Mesh_Record_Component.SCALAR]
    component.position = position
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)

    series.flush()
    del series
