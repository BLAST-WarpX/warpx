#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c

# Read in the cross-section data for comparison with theory
cross_section_data = np.loadtxt(
    "../../../../warpx-data/MCC_cross_sections/H/H_on_H2_ionization.dat"
)
E_eV = 12e3
E_com_eV = E_eV * 2.0 / 3
# Find the cross-section at the center of mass energy
cross_section = np.interp(E_com_eV, cross_section_data[:, 0], cross_section_data[:, 1])
print(f"Cross-section at {E_com_eV} eV: {cross_section} m^2")

# Read in the data
ts = OpenPMDTimeSeries("./diags/diag")

# Compute the beam flux for ions and neutral, as a function of z, and compute with theory
iteration = 100
sim_flux = {"Hneutral": [], "Hplus": []}
theory_flux = {"Hneutral": [], "Hplus": []}

# Compute the simulation flux
# Loop over species
for species in ["Hneutral", "Hplus"]:
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
zmax = 0.2
z_th = bins[:-1]
theory_flux["Hneutral"] = flux * np.exp(-z_th * n * cross_section)
theory_flux["Hplus"] = flux * (1 - np.exp(-z_th * n * cross_section))

# Compare the fluxes
assert np.allclose(sim_flux["Hneutral"], theory_flux["Hneutral"], atol=5e-2 * flux)
assert np.allclose(sim_flux["Hplus"], theory_flux["Hplus"], atol=5e-2 * flux)

# Plot the computed fluxes
plt.figure()
for species, color in zip(["Hneutral", "Hplus"], ["b", "r"]):
    plt.plot(bins[:-1], sim_flux[species], color=color, label=species)
    plt.plot(z_th, theory_flux[species], color=color, ls=":")
plt.legend(loc=0)
plt.ylabel("Beam flux")
plt.xlabel("z [m]")
plt.savefig("Beam_fluxes.png")
