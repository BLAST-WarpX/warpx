#!/usr/bin/env python3


import sys

import matplotlib.pyplot as plt
import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, epsilon_0, m_u

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)
# Parameters used in the corresponding input script
kV = 1000
cm = 0.01
extractor_voltage = -93.0 * kV
d_plate = 8.0 * cm
ion_mass = 39 * m_u

it = ts.iterations[-1]
phi, meta = ts.get_field("phi", iteration=it, plot=False)
ez, _ = ts.get_field("E", "z", iteration=ts.iterations[-1], plot=False)
rho, _ = ts.get_field("rho", iteration=it, plot=False)
jz, _ = ts.get_field("j", "z", iteration=it, plot=False)
z, uz = ts.get_particle(["z", "uz"], iteration=it)
time_cur = ts.current_t

# Calculate theoretical Child-Langmuir limit for a given voltage
jz_CL_theory = (
    (4 / 9)
    * epsilon_0
    * np.sqrt(2 * abs(e) / ion_mass)
    * abs(extractor_voltage) ** (3 / 2)
    / d_plate**2
)
phi_CL_theory = extractor_voltage * (meta.z / d_plate) ** (4 / 3)
ez_CL_theory = -(4 / (3 * d_plate)) * extractor_voltage * (meta.z / d_plate) ** (1 / 3)
rho_CL_theory = (
    epsilon_0
    * (4 / (3 * d_plate) ** 2)
    * extractor_voltage
    * (meta.z / d_plate) ** (-2 / 3)
)
uz_CL_theory = -jz_CL_theory / rho_CL_theory / c

color = "orange"
title = r"$\Gamma_{ions}=38.79 \approx \Gamma_{CL}$"

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(f"{title}\n time = {np.round(time_cur / 1e-6, 5)} $\\mu s$")
axs[0, 0].scatter(z, uz, color=color, label=title, s=0.2)
axs[0, 0].plot(meta.z, uz_CL_theory, ls=":", color="black")
axs[0, 0].set_title(r"$u_z$")
axs[0, 0].set_xlabel("z, mm")

axs[0, 1].plot(meta.z, ez, color=color, label="WarpX ")
axs[0, 1].set_title(r"$E_z$")
axs[0, 1].set_xlabel("z, mm")
axs[0, 1].plot(
    meta.z, ez_CL_theory, label="Child-Langmuir limit (theory)", ls=":", color="black"
)
axs[0, 1].legend()

axs[1, 0].plot(meta.z, jz, color=color)
axs[1, 0].set_title(r"$J_z$")
axs[1, 0].axhline(y=jz_CL_theory, color="black", linestyle="-")
axs[1, 0].set_xlabel("z, mm")

axs[1, 1].plot(meta.z, phi, color=color)
axs[1, 1].set_title(r"$\phi$")
axs[1, 1].set_xlabel("z, mm")
axs[1, 1].plot(meta.z, phi_CL_theory, label="theory", ls=":", color="black")

plt.tight_layout()

rel_error_phi = np.abs(phi[1:] - phi_CL_theory[1:]) / np.abs(phi_CL_theory[1:])
rel_error_jz = np.abs(jz - jz_CL_theory) / jz_CL_theory
tolerance = 0.2

assert np.all(rel_error_jz < tolerance) and np.all(rel_error_phi < tolerance), (
    "Child–Langmuir limit is violated! "
)
