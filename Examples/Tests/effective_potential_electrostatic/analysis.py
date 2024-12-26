#!/usr/bin/env python3
#
# --- Analysis script for the effective potential Poisson solver test. This test
# --- is based on the adiabatic plasma expansion benchmark from Connor et al. (2021)
# --- doi.org/10.1109/TPS.2021.3072353.
# --- The electron density distribution (as a function of radius) is compared
# --- with the analytically calculated density based on the input parameters
# --- of the test simulation at each output timestep.

import dill
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.interpolate import RegularGridInterpolator

from pywarpx import picmi

constants = picmi.constants

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

# characteristic expansion time
tau = sim["sigma_0"] * np.sqrt(sim["M"] / (constants.kb * (sim["T_e"] + sim["T_i"])))


def get_analytic_density(r, t):
    expansion_factor = 1.0 + t**2 / tau**2
    T = sim["T_e"] / expansion_factor
    sigma = sim["sigma_0"] * np.sqrt(expansion_factor)
    return (
        sim["n_plasma"] * (T / sim["T_e"]) ** 1.5 * np.exp(-(r**2) / (2.0 * sigma**2))
    )


def get_radial_function(field, info):
    """Helper function to transform Cartesian data to polar data and average
    over theta and phi."""
    r_grid = np.linspace(0, np.max(info.z) - 1e-3, 30)
    theta_grid = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    phi_grid = np.linspace(0, np.pi, 20)

    r, t, p = np.meshgrid(r_grid, theta_grid, phi_grid, indexing="ij")
    x_sp = r * np.sin(p) * np.cos(t)
    y_sp = r * np.sin(p) * np.sin(t)
    z_sp = r * np.cos(p)

    interp = RegularGridInterpolator((info.x, info.y, info.z), field)
    field_sp = interp((x_sp, y_sp, z_sp))
    return r_grid, np.mean(field_sp, axis=(1, 2))


diag_dir = "diags/field_diag"
ts = OpenPMDTimeSeries(diag_dir, check_all_files=True)

rms_errors = np.zeros(len(ts.iterations))

for ii, it in enumerate(ts.iterations):
    rho_e, info = ts.get_field(field="rho_electrons", iteration=it)
    r_grid, n_e = get_radial_function(-rho_e / constants.q_e, info)

    n_e_analytic = get_analytic_density(r_grid, ts.t[ii])
    rms_errors[ii] = (
        np.sqrt(np.mean(np.sum((n_e - n_e_analytic) ** 2))) / n_e_analytic[0]
    )

    plt.plot(r_grid, n_e_analytic, "k--", alpha=0.6)
    plt.plot(r_grid, n_e, label=f"t = {ts.t[ii]*1e6:.2f} $\mu$s")

print("RMS error (%) in density: ", rms_errors)
assert np.all(rms_errors < 0.05)

plt.ylabel("$n_e$ (m$^{-3}$)")
plt.xlabel("r (m)")
plt.grid()
plt.legend()
plt.show()
