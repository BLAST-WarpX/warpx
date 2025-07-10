#!/usr/bin/env python3


import sys

import matplotlib.pyplot as plt
import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import e, epsilon_0, m_u

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]
print("-----", filename)
ts = OpenPMDTimeSeries(filename)
# Parameters used in the corresponding input script
kV = 1000
cm = 0.01
extractor_voltage = -93.0 * kV
d_plate = 8.0 * cm
ion_mass = 39 * m_u

# Calculate Child-Langmuir limit for a given voltage
J_CL = (
    (4 / 9)
    * epsilon_0
    * np.sqrt(2 * abs(e) / ion_mass)
    * abs(extractor_voltage) ** (3 / 2)
    / d_plate**2
)
print(ts.iterations[-1])
it = ts.iterations[-1]
phi, meta = ts.get_field("phi", iteration=it, plot=False)

phi_analyt = extractor_voltage * (meta.z / d_plate) ** (4 / 3)
ez, _ = ts.get_field("E", "z", iteration=ts.iterations[-1], plot=False)
rho, _ = ts.get_field("rho", iteration=it, plot=False)
jz, _ = ts.get_field("j", "z", iteration=it, plot=False)
z, uz = ts.get_particle(["z", "uz"], iteration=it)
time_cur = ts.current_t
color = "orange"
title = r"$\Gamma_{ions}=38.79 \approx \Gamma_{CL}$"

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("time = " + str(np.round(time_cur / 1e-6, 5)) + r" $ \mu s$")
axs[0, 0].scatter(z, uz, color=color, label=title, s=0.2)
axs[0, 0].set_title(r"$u_z$")
axs[0, 0].set_xlabel("z, mm")

axs[0, 1].plot(meta.z, ez, color=color, label=title)
axs[0, 1].set_title(r"$E_z$")
axs[0, 1].set_xlabel("z, mm")
axs[0, 1].legend()

axs[1, 0].plot(meta.z, jz, color=color)
axs[1, 0].set_title(r"$J_z$")
axs[1, 0].axhline(
    y=J_CL, color="black", linestyle="-", label=r"$y = J_{CL}$ (Child-Langmuir limit)"
)
axs[1, 0].set_xlabel("z, mm")
axs[1, 0].legend(loc="upper left")

axs[1, 1].plot(meta.z, phi, color=color)  # , label='WarpX')
axs[1, 1].set_title(r"$phi$")
axs[1, 1].set_xlabel("z, mm")
axs[1, 1].plot(meta.z, phi_analyt, label="theory", ls=":", color="black")
axs[1, 1].legend()
plt.tight_layout()

rel_error_phi = np.abs(phi[1:] - phi_analyt[1:]) / np.abs(phi_analyt[1:])
rel_error_jz = np.abs(jz - J_CL) / J_CL
tolerance = 0.2

assert np.all(rel_error_jz < tolerance) and np.all(rel_error_phi < tolerance), (
    "Child–Langmuir limit is violated! "
)
