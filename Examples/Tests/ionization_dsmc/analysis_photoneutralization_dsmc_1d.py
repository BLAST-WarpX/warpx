#!/usr/bin/env python3

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c as clight

iteration = 100

sim_mass = {"Hminus": [], "electrons": [], "Hneutral": []}
sim_uz = {"Hminus": [], "photons": [], "electrons": [], "Hneutral": []}
sim_weight = {"Hminus": [], "photons": [], "electrons": [], "Hneutral": []}

theory_uz = {"electrons": [], "Hneutral": []}

# Read in the data
ts = OpenPMDTimeSeries("./diags/diag")
ds = OpenPMDTimeSeries("./diags/bound/particles_at_zlo/")

for species in ["Hminus", "photons", "Hneutral"]:
    sim_uz[species], sim_mass[species], sim_weight[species] = ts.get_particle(
        species=species, var_list=["uz", "mass", "w"], iteration=iteration
    )

# Note: Electrons move so fast that they exit the simulation box immediately after the reaction occurs.
# Therefore, we monitor them using the boundary scraping diagnostic.
sim_uz["electrons"], sim_mass["electrons"], sim_weight["electrons"] = ds.get_particle(
    species="electrons", var_list=["uz", "mass", "w"], iteration=iteration
)

# Calculate theoretical velocities for Hneutral and electrons species for a given Q = -0.756 eV
eV_to_J = 1.602e-19
m_H = 1.6726e-27  # [kg]
m_e = 9.109e-31  # [kg], electron mass

E0 = 1.2  # [eV], initial kinetic energy of Hminus
Q = -0.756  # [eV], reaction energy cost

# Compute initial velocities
vminus = np.sqrt(2 * E0 * eV_to_J / m_H)  # [m/s] initial Hminus velocity

# Photon energy
uz_photon = -1.9569471624266145e-06  # normalized by m_e*c^2
E_photon = abs(uz_photon) * m_e * clight**2  # [J]
p_photon = (uz_photon * m_e * clight**2) / clight  # [kg*m/s]

# Compute center-of-mass (COM) quantities
v_com = (m_H * vminus + p_photon) / (m_H + m_e)

E_com_before = (
    0.5 * m_H * (vminus - v_com) ** 2 + E_photon - p_photon * v_com
) / eV_to_J  # [eV]

E_com_after = (
    E_com_before + Q
)  # [eV] Total kinetic energy in the center-of-mass frame (after the reaction)

# Compute velocity of Hneutral in COM frame
vH_com = np.sqrt(
    2 * (E_com_after * eV_to_J) / (m_H * (1 + m_H / m_e))
)  # [m/s] Note: in the COM frame, the total momentum is zero
vE_com = -m_H / m_e * vH_com  # [m/s] electrons velocity in COM frame

# Lab-frame velocities
vH_lab = v_com + vH_com  # Hneutral
vE_lab = v_com + vE_com  # electrons

theory_uz["Hneutral"] = vH_lab / clight
theory_uz["electrons"] = vE_lab / clight

# Compare the velocities
assert np.allclose(sim_uz["electrons"], theory_uz["electrons"], rtol=1e-1)
assert np.allclose(sim_uz["Hneutral"], theory_uz["Hneutral"], rtol=1e-3)
