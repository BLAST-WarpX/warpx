#!/usr/bin/env python3

# Copyright 2019-2023 Grant Johnson, Remi Lehe
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs_1d`. This simulates a 1D WFA with Pondermotive Envelope:
# REF: (Equations 20-23) https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.81.1229
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(50)

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

# this will be the name of the plot file
fn = sys.argv[1]

# Parameters (these parameters must match the parameters in `inputs.multi.rt`)
n0 = 20.0e23
# Plasma frequency
wp = np.sqrt((n0 * e**2) / (m_e * epsilon_0))
kp = wp / c
tau = 15.0e-15
a0 = 2.491668
e = -e  # Electrons
lambda_laser = 0.8e-6

zmin = -20e-6
zmax = 100.0e-6
Nz = 4864

# Compute the theory

# Call the ode solver
from scipy.integrate import odeint


# ODE Function
def odefcn(phi, xi, kp, a0, c, tau, xi_0, lambda_laser):
    phi1, phi2 = phi
    a_sq = (
        a0**2
        * np.exp(-2 * (xi - xi_0) ** 2 / (c**2 * tau**2))
        * np.sin(2 * np.pi * (xi - xi_0) / lambda_laser) ** 2
    )
    dphi1_dxi = phi2
    dphi2_dxi = kp**2 * ((1 + a_sq) / (2 * (1 + phi1) ** 2) - 0.5)
    return [dphi1_dxi, dphi2_dxi]


# Call odeint to solve the ODE
xi_span = [-20e-6, 100e-6]
xi_0 = 0e-6
phi0 = [0.0, 0.0]
dxi = (zmax - zmin) / Nz
xi = zmin + dxi * (0.5 + np.arange(Nz))
phi = odeint(odefcn, phi0, xi, args=(kp, a0, c, tau, xi_0, lambda_laser))

# Change array direction to match the simulations
xi = -xi[::-1]
phi = phi[::-1]
xi_0 = -0e-6
phi2 = phi[:, 0]
Ez = -phi[:, 1]

# Compute the derived quantities
a_sq = (
    a0**2
    * np.exp(-2 * (xi - xi_0) ** 2 / (c**2 * tau**2))
    * np.sin(2 * np.pi * (xi - xi_0) / lambda_laser) ** 2
)
gamma_perp_sq = 1 + a_sq
n = n0 * (gamma_perp_sq + (1 + phi2) ** 2) / (2 * (1 + phi2) ** 2)
uz = (gamma_perp_sq - (1 + phi2) ** 2) / (2 * (1 + phi2))
gamma = (gamma_perp_sq + (1 + phi2) ** 2) / (2 * (1 + phi2))

# Theory Components [convert to si]
uz *= c
J_th = np.multiply(np.divide(uz, gamma), n)
J_th *= e
rho_th = e * n
E_th = Ez
E_th *= (m_e * c * c) / e
V_th = np.divide(uz, gamma)
V_th /= c
# Remove the ions
rho_th = rho_th - e * n0

# Dictate which region to compare solutions over (cuttoff 0's from BTD extra)
min_i = 200
max_i = 4864

# Read the file
ds = yt.load(fn)
t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
# Check the validity of the fields
error_rel = 0
for field in ["Ez"]:
    E_sim = data[("mesh", field)].to_ndarray()[:, 0, 0]
    # E_th = get_theoretical_field(field, t0)
    max_error = (
        abs(E_sim[min_i:max_i] - E_th[min_i:max_i]).max() / abs(E_th[min_i:max_i]).max()
    )
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)

# Check the validity of the currents
for field in ["Jz"]:
    J_sim = data[("mesh", field)].to_ndarray()[:, 0, 0]
    # J_th = get_theoretical_J_field(field, t0)
    max_error = (
        abs(J_sim[min_i:max_i] - J_th[min_i:max_i]).max() / abs(J_th[min_i:max_i]).max()
    )
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)

# Check the validity of the charge
for field in ["rho"]:
    rho_sim = data[("boxlib", field)].to_ndarray()[:, 0, 0]
    # rho_th = get_theoretical_rho_field(field, t0)
    max_error = (
        abs(rho_sim[min_i:max_i] - rho_th[min_i:max_i]).max()
        / abs(rho_th[min_i:max_i]).max()
    )
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)

V_sim = np.divide(J_sim, rho_sim)
V_sim /= c

# Create a figure with 2 rows and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Titles and labels
titles = ["Ez", "rho", "Jz", "Vz/c"]
xlabel = r"Xi"
ylabel = ["Ez", "rho", "Jz", "Vz/c"]

# Plotting loop
for i in range(3):
    ax = axes[i // 2, i % 2]  # Get the current subplot

    # Plot theoretical data
    ax.plot(xi, [E_th, rho_th, J_th, V_th][i], label="Theoretical")

    # Plot simulated data
    ax.plot(xi, [E_sim, rho_sim, J_sim, V_sim][i], label="Simulated")

    # Set titles and labels
    ax.set_title(f"{titles[i]} vs Xi")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel[i])

    # Add legend
    ax.legend()

# Adjust subplot layout
plt.tight_layout()

# Save the figure
plt.savefig("wfa_fluid_nonlinear_1d_analysis.png")

plt.show()


tolerance_rel = 0.30

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
