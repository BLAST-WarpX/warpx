#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script checks the momentum distribution of particles whose temperature
# (in eV) is read from an openPMD file, in RZ, RCYLINDER or RSPHERE geometry.
# The file is created by inputs_base_temperature_from_file_prepare.py.
# For each species, the momenta of the particles, normalized by the expected
# thermal spread at the particle position, should follow a standard normal
# distribution. This is checked separately in several radial bins, so that
# an incorrect radial dependence of the temperature would be detected.

import argparse

import numpy as np
from inputs_base_temperature_from_file_prepare import temperature_in_eV
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, m_e

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=["rz", "rcylinder", "rsphere"], required=True)
args = parser.parse_args()

tolerance = 2e-2
n_radial_bins = 4

ts = OpenPMDTimeSeries("./diags/diag1")

for species in ["maxwellian", "maxwell_juttner"]:
    print(f"Species: {species}")
    if args.geometry == "rcylinder":
        # RCYLINDER does not output the z position
        x, y, ux, uy, uz = ts.get_particle(
            ["x", "y", "ux", "uy", "uz"], species=species, iteration=0
        )
        r = np.sqrt(x**2 + y**2)
        T_eV = temperature_in_eV(r)
    else:
        x, y, z, ux, uy, uz = ts.get_particle(
            ["x", "y", "z", "ux", "uy", "uz"], species=species, iteration=0
        )
        if args.geometry == "rz":
            r = np.sqrt(x**2 + y**2)
            T_eV = temperature_in_eV(r, z)
        else:
            r = np.sqrt(x**2 + y**2 + z**2)
            T_eV = temperature_in_eV(r)

    # Expected thermal spread of each momentum component
    u_std = np.sqrt(T_eV * e / (m_e * c**2))

    r_edges = np.linspace(0.0, r.max(), n_radial_bins + 1)
    for ibin in range(n_radial_bins):
        in_bin = (r >= r_edges[ibin]) & (r <= r_edges[ibin + 1])
        for u_name, u in [("ux", ux), ("uy", uy), ("uz", uz)]:
            u_normalized = u[in_bin] / u_std[in_bin]
            mean = np.mean(u_normalized)
            std = np.std(u_normalized)
            print(
                f"  {r_edges[ibin]:.2f} <= r <= {r_edges[ibin + 1]:.2f}, {u_name}: "
                f"{in_bin.sum()} particles, mean = {mean:.4f}, std = {std:.4f}"
            )
            assert abs(mean) < tolerance
            assert abs(std - 1.0) < tolerance
