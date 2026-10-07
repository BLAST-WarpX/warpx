#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c

# Read in the cross-section data for comparison with theory
cross_section_data = np.loadtxt(
    "../../../../warpx-data/MCC_cross_sections/H/Hion_on_H2_charge_exchange.dat"
)
E_eV = 12e3
E_com_eV = E_eV * 2.0 / 3
# Find the cross-section at the center of mass energy
cross_section = np.interp(E_com_eV, cross_section_data[:, 0], cross_section_data[:, 1])
print(f"Cross-section at {E_com_eV} eV: {cross_section} m^2")

# Number of cells per collision bin (optional argument, default 1),
# i.e. the value of `bin_size_ratio` used in the simulation
cells_per_bin = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Read in the data
ts = OpenPMDTimeSeries("./diags/diag")

# Compute the beam flux for ions and neutral, as a function of z, and compute with theory
iteration = 100
sim_flux = {"H": [], "Hplus": []}
theory_flux = {"H": [], "Hplus": []}

# Compute the simulation flux
# Loop over species
for species in ["H", "Hplus"]:
    z, w, uz = ts.get_particle(
        ["z", "w", "uz"],
        species=species,
        iteration=iteration,
        select={"uz": [1e-3, None]},
    )
    w_binned, bins = np.histogram(z, bins=32, range=[0, 0.2], weights=w * uz)
    # Convert from number of particles per bin to beam current
    dz_bin = bins[1] - bins[0]
    sim_flux[species] = w_binned / dz_bin * c

# Compute the theoretical flux
n = 1e21
flux = 1e20
z_th = bins[:-1]
theory_flux["Hplus"] = flux * np.exp(-z_th * n * cross_section)  # remaining Hplus flux
theory_flux["H"] = flux * (
    1 - np.exp(-z_th * n * cross_section)
)  # H flux, which underwent the charge exchange

# Compare the fluxes
assert np.allclose(sim_flux["Hplus"], theory_flux["Hplus"], atol=5e-2 * flux)
if cells_per_bin == 1:
    assert np.allclose(sim_flux["H"], theory_flux["H"], atol=5e-2 * flux)
else:
    # The H products of charge exchange are created at the position of the H2 reactant,
    # which can be anywhere within the collision bin. The H flux is thus only resolved at
    # the scale of the collision bins: compare the fluxes averaged over the collision bins.
    # The first collision bin is excluded, since some of the H created there (where the
    # flux varies the most) end up downstream.
    n_coarse = len(z_th) // cells_per_bin
    n_fine = n_coarse * cells_per_bin
    sim_coarse = sim_flux["H"][:n_fine].reshape(n_coarse, cells_per_bin).mean(axis=1)
    theory_coarse = (
        theory_flux["H"][:n_fine].reshape(n_coarse, cells_per_bin).mean(axis=1)
    )
    assert np.allclose(sim_coarse[1:], theory_coarse[1:], atol=5e-2 * flux)

# Plot the computed fluxes
plt.figure()
for species, color in zip(["H", "Hplus"], ["b", "r"]):
    plt.plot(bins[:-1], sim_flux[species], color=color, label=species)
    plt.plot(z_th, theory_flux[species], color=color, ls=":")
plt.legend(loc=0)
plt.ylabel("Beam flux")
plt.xlabel("z [m]")
plt.savefig("Beam_fluxes.png")
