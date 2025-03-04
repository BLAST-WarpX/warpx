#!/usr/bin/env python3
# DSMC ionization test script:
#   - compares WarpX simulation results with theoretical model predictions.

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from openpmd_viewer import OpenPMDTimeSeries
from scipy.stats import qmc

from pywarpx import picmi

constants = picmi.constants

ts = OpenPMDTimeSeries("diags/diag2/")

q_e = constants.q_e
m_p = constants.m_p
m_e = constants.m_e
k_B = constants.kb
ep0 = constants.ep0
clight = constants.c

plasma_density = 1e14
neutral_density = 1e20
dt = 1e-9
electron_temp = 10
neutral_temp = 300
max_steps = 250
max_time = max_steps * dt

L = [0.1] * 3

sigma_iz_file = "../../../../warpx-data/MCC_cross_sections/Xe/ionization.dat"
iz_data = np.loadtxt(sigma_iz_file)

energy_eV = iz_data[:, 0]
sigma_m2 = iz_data[:, 1]
iz_energy = energy_eV[0]


def get_Te(ts):
    T_e = []
    for iteration in tqdm.tqdm(ts.iterations):
        ux, uy, uz = ts.get_particle(
            ["ux", "uy", "uz"], species="electrons", iteration=iteration
        )
        v_std_x = np.std(ux * clight)
        v_std_y = np.std(uy * clight)
        v_std_z = np.std(uz * clight)
        v_std = (v_std_x + v_std_y + v_std_z) / 3
        T_e.append(m_e * v_std**2 / q_e)
    return T_e


def get_density(ts):
    number_data = np.genfromtxt("diags/counts.txt")
    Te = get_Te(ts)
    total_volume = L[0] * L[1] * L[2]
    electron_weight = number_data[:, 8]
    neutral_weight = number_data[:, 9]
    ne = electron_weight / total_volume
    nn = neutral_weight / total_volume
    return [ne, nn, ne * Te]


def compute_rate_coefficients(temperatures_eV, energy_eV, sigma_m2, num_samples=1024):
    """Integrate cross sections over maxwellian VDF to obtain reaction rate coefficients
    Given electron energy in eV (`energy_eV`) and reaction cross sections at those energies (`sigma_m2`),
    this function computes the reaction rate coefficient $k(T_e)$ for maxwellian electrons at
    a provided list of electron temperatures `temperatures_eV`.

    The rate coefficient is given by

    $$
    k(T_e) = \int \sigma(E) E dE
    $$

    where the energy $E$ is drawn from a maxwellian distribution function with zero speed and temperature $T_e$.
    We solve this using a quasi-monte carlo approach, by drawing a large number of low-discrepancy samples from
    the appropriate distribution and obtaining the average of $\sigma(E) E$.
    """
    thermal_speed_scale = np.sqrt(q_e / m_e)
    k = np.zeros(temperatures_eV.size)

    # obtain low-discrepancy samples of normal dist
    dist = qmc.MultivariateNormalQMC(np.zeros(3), np.eye(3))
    v = dist.random(num_samples)

    for i, T in enumerate(temperatures_eV):
        # scale velocities to proper temperature
        # compute energies corresponding to each sampled velocity vector
        speed_squared = (v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2) * T
        e = 0.5 * speed_squared
        speed = np.sqrt(speed_squared) * thermal_speed_scale
        # get cross section by interpolating on table
        sigma = np.interp(e, energy_eV, sigma_m2, left=0)
        k[i] = np.mean(sigma * speed)
    return k


def rhs(state, params):
    """Compute the right-hand side of ODE system that solves the global model described below.
    The global model solves for the evolution of plasma density ($n_e$), neutral density ($n_n),
    and electron temperature ($T_e$) in the presence of ionization.
    The model equations consist of a continuity equation for electrons and neutrals,
    combined with an energy equation for electrons.

    $$
    \frac{\partial n_e}{\partial t} = \dot{n}
    \frac{\partial n_n}{\partial t} = -\dot{n}
    \frac{3}{2}\frac{\partial n_e T_e}{\partial t} = -\dot{n} \epsilon_{iz},
    $$

    where

    $$
    \dot{n} = n_n n_e k_{iz}(T_e),
    $$

    $k_iz$ is the ionization rate coefficient as a function of electron temperature in eV, and $\epsilon_{iz}$ is the ionization energy cost in eV.
    """
    # unpack parameters
    E_iz, Te_table, kiz_table = params
    ne, nn, energy = state[0], state[1], state[2]

    # compute rhs
    Te = energy / ne / 1.5
    rate_coeff = np.interp(Te, Te_table, kiz_table, left=0)
    ndot = ne * nn * rate_coeff
    f = np.empty(3)

    # fill in ionization rate
    f[0] = ndot  # d(ne)/dt
    f[1] = -ndot  # d(nn)/dt
    f[2] = -ndot * iz_energy  # -d(ne*eps) / dt
    return f


def solve_theory_model():
    # integrate cross-section table to get rate coefficients
    Te_table = np.linspace(0, 2 * electron_temp, 256)
    kiz_table = compute_rate_coefficients(
        Te_table, energy_eV, sigma_m2, num_samples=4096
    )

    # set up system
    num_steps = max_steps + 1
    state_vec = np.zeros((num_steps, 3))
    state_vec[0, :] = np.array(
        [plasma_density, neutral_density, 1.5 * plasma_density * electron_temp]
    )
    t = np.linspace(0, max_time, num_steps)
    params = (iz_energy, Te_table, kiz_table)

    # solve the system (use RK2 integration)
    for i in range(1, num_steps):
        u = state_vec[i - 1, :]
        k1 = rhs(u, params)
        k2 = rhs(u + k1 * dt, params)
        state_vec[i, :] = u + (k1 + k2) * dt / 2

    # return result
    ne = state_vec[:, 0]
    nn = state_vec[:, 1]
    Te = state_vec[:, 2] / (1.5 * ne)
    return t, [ne, nn, ne * Te]


t_warpx = np.loadtxt("diags/counts.txt")[:, 1]
data_warpx = get_density(ts)

t_theory, data_theory = solve_theory_model()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1
method = "dsmc"
labels = ["$n_e$ [m$^{-3}$]", "$n_n$ [m${-3}$]", "$n_e T_e$ [eVm$^{-3}$]"]
titles = ["Plasma density", "Neutral density", "Normalized electron temperature"]

for i, (title, label, field_warpx, field_theory) in enumerate(
    zip(titles, labels, data_warpx, data_theory)
):
    axs[i].set_ylabel(label)
    axs[i].set_title(title)
    axs[i].plot(t_warpx, field_warpx, label="WarpX (" + method + ")")
    axs[i].plot(t_theory, field_theory, label="theory", color="red", ls="--")

    axs[i].legend()
plt.tight_layout()
plt.savefig("ionization_dsmc_density_Te.png", dpi=150)


tolerances = [4e-2, 1e-6, 4e-2]


def check_tolerance(array, tolerance):
    assert np.all(array <= tolerance), (
        f"Test did not pass: one or more elements exceed the tolerance of {tolerance}."
    )
    print("All elements of are within the tolerance.")


plt.figure()
labels = [
    "Plasma density $(n_e$)",
    "Neutral density $(n_n$)",
    "Normalized electron temperature $(T_en_e$)",
]
for i, (label, field_warpx, field_theory, tolerance) in enumerate(
    zip(labels, data_warpx, data_theory, tolerances)
):
    relative_error = np.array(
        abs((data_warpx[i] - data_theory[i][::5]) / data_theory[i][::5])
    )
    plt.plot(t_warpx, relative_error, label=label)
    plt.ylabel("Relative error")
    plt.xlabel("Time [s]")
    plt.legend()
    check_tolerance(relative_error, tolerance)
plt.savefig("./relative_error_density_Te.png", dpi=150)
