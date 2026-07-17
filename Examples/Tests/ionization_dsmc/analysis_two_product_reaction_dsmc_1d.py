#!/usr/bin/env python3

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c as clight

iteration = 100

sim_uz = {"H": [], "H2plus": []}
theory_uz = {"H": [], "H2plus": []}

# Read in the data
ts = OpenPMDTimeSeries("./diags/diag")

# Get H and H2+ species velocities from the simulation
sim_uz["H"] = ts.get_particle(species="H", var_list=["uz"], iteration=iteration)[0]
sim_uz["H2plus"] = ts.get_particle(
    species="H2plus", var_list=["uz"], iteration=iteration
)[0]

# Calculate theoretical velocities for H and H2+ species for a given Q = -1.66 eV
eV_to_J = 1.602e-19
m_H = 1.6726e-27  # [kg]
m_H2 = 2 * m_H  # [kg]

E0 = 5.0  # [eV], initial kinetic energy of H+
Q = -1.66  # [eV], reaction energy cost

# Compute initial velocities
v0 = np.sqrt(2 * E0 * eV_to_J / m_H)  # [m/s] initial H+ velocity
v_H2 = 0.0  # [m/s] initial H2 velocity

# Compute center-of-mass (COM) quantities
v_com = (m_H * v0 + m_H2 * v_H2) / (m_H + m_H2)
E_com_before = (
    m_H2 / (m_H + m_H2)
) * E0  # [eV] Total kinetic energy in the center-of-mass frame (before the reaction)
# (e.i., E_com_before  = ( (m_H * m_H2 / (m_H + m_H2) ) * (v0-v_H2)**2 )/2)
E_com_after = (
    E_com_before + Q
)  # [eV] Total kinetic energy in the center-of-mass frame (after the reaction)

# Compute velocity of H in COM frame
vH_com = np.sqrt(
    2 * (E_com_after * eV_to_J) / (m_H * (1 + m_H / m_H2))
)  # [m/s] Note: in the COM frame, the total momentum is zero

# Lab-frame velocities
vH_lab = v_com + vH_com
vH2_lab = v_com - vH_com / 2

theory_uz["H"] = vH_lab / clight
theory_uz["H2plus"] = vH2_lab / clight

# Compare the velocities
assert np.allclose(sim_uz["H2plus"], theory_uz["H2plus"], atol=1e-8)
assert np.allclose(sim_uz["H"], theory_uz["H"], atol=1e-8)
