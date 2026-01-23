#!/usr/bin/env python3
#
# --- Analysis script for the hybrid-PIC example producing EM modes.

import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pywarpx import picmi

constants = picmi.constants

matplotlib.rcParams.update({"font.size": 20})

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

if sim.B_dir == "z":
    field_idx_dict = {"z": 4, "Ez": 7, "Bx": 8, "By": 9}
    data = np.loadtxt("diags/par_field_data.txt", skiprows=1)
else:
    if sim.dim == 1:
        field_idx_dict = {"z": 4, "Ez": 7, "Bx": 8, "By": 9}
    else:
        field_idx_dict = {"z": 2, "Ez": 3, "Bx": 4, "By": 5}
    data = np.loadtxt("diags/perp_field_data.txt", skiprows=1)

# step, t, z, Ez, Bx, By = raw_data.T
step = data[:, 0]

num_steps = len(np.unique(step))

# get the spatial resolution
resolution = len(np.where(step == 0)[0]) - 1

# reshape to separate spatial and time coordinates
sim_data = data.reshape((num_steps, resolution + 1, data.shape[1]))

z_grid = sim_data[1, :, field_idx_dict["z"]]
idx = np.argsort(z_grid)[1:]
dz = np.mean(np.diff(z_grid[idx]))
dt = np.mean(np.diff(sim_data[:, 0, 1]))

data = np.zeros((num_steps, resolution, 3))
for i in range(num_steps):
    data[i, :, 0] = sim_data[i, idx, field_idx_dict["Bx"]]
    data[i, :, 1] = sim_data[i, idx, field_idx_dict["By"]]
    data[i, :, 2] = sim_data[i, idx, field_idx_dict["Ez"]]

print(f"Data file contains {num_steps} time snapshots.")
print(f"Spatial resolution is {resolution}")


def get_analytic_R_mode(w):
    return w / np.sqrt(1.0 + abs(w))


def get_analytic_L_mode(w):
    print(w, w / np.sqrt(1.0 - abs(w)))
    return w / np.sqrt(1.0 - abs(w))


if sim.B_dir == "z":
    global_norm = (
        1.0
        / (2.0 * constants.mu0)
        / ((3.0 / 2) * sim.n_plasma * sim.T_plasma * constants.q_e)
    )
else:
    global_norm = (
        constants.ep0 / 2.0 / ((3.0 / 2) * sim.n_plasma * sim.T_plasma * constants.q_e)
    )

if sim.B_dir == "z":
    Bl = (data[:, :, 0] + 1.0j * data[:, :, 1]) / np.sqrt(2.0)
    field_kw = np.fft.fftshift(np.fft.fft2(Bl))
else:
    field_kw = np.fft.fftshift(np.fft.fft2(data[:, :, 2]))

w_norm = sim.w_ci
if sim.B_dir == "z":
    k_norm = 1.0 / sim.l_i
else:
    k_norm = 1.0 / sim.rho_i

k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(resolution, dz)) / k_norm
w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(num_steps, dt)) / w_norm
w = -np.flipud(w)

# aspect = (xmax-xmin)/(ymax-ymin) / aspect_true
extent = [k[0], k[-1], w[0], w[-1]]

fig, ax1 = plt.subplots(1, 1, figsize=(10, 7.25))

if sim.B_dir == "z" and sim.dim == 1:
    vmin = -1 if sim.test else 1.5
    vmax = None if sim.test else 5.0
else:
    vmin = -2.75  # None # 6
    vmax = 3.25  # None # 3

im = ax1.imshow(
    np.log10(np.abs(field_kw**2) * global_norm),
    extent=extent,
    aspect="equal",
    cmap="inferno",
    vmin=vmin,
    vmax=vmax,
)

# Colorbars
fig.subplots_adjust(right=0.5)
cbar_ax = fig.add_axes([0.525, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax, orientation="vertical")

# cbar_lab = r'$\log_{10}(\frac{|B_{R/L}|^2}{2\mu_0}\frac{2}{3n_0k_BT_e})$'
if sim.B_dir == "z":
    cbar_lab = r"$\log_{10}(\beta_{R/L})$"
else:
    cbar_lab = r"$\log_{10}(\varepsilon_0|E_z|^2/(3n_0k_BT_e))$"
cbar_ax.set_ylabel(cbar_lab, rotation=270, labelpad=30)

if sim.B_dir == "z":
    # plot the L mode
    idx = np.where(w > 0)[0]
    ax1.plot(
        get_analytic_L_mode(w[idx]),
        np.abs(w[idx]),
        c="limegreen",
        ls="--",
        lw=1.25,
        label="L mode:\n" + r"$(kl_i)^2=\frac{(\omega/\Omega_i)^2}{1-\omega/\Omega_i}$",
    )
    # plot the R mode
    idx = np.where(w < 0)[0]
    ax1.plot(
        get_analytic_R_mode(w[idx]),
        -np.abs(w[idx]),
        c="limegreen",
        ls="-.",
        lw=1.25,
        label="R mode:\n" + r"$(kl_i)^2=\frac{(\omega/\Omega_i)^2}{1+\omega/\Omega_i}$",
    )

    ax1.plot(
        k,
        1.0 + 3.0 * sim.v_ti / w_norm * k * k_norm,
        c="limegreen",
        ls=":",
        lw=1.25,
        label=r"$\omega = \Omega_i + 3v_{th,i} k$",
    )
    ax1.plot(
        k, 1.0 - 3.0 * sim.v_ti / w_norm * k * k_norm, c="limegreen", ls=":", lw=1.25
    )

    ax1.plot(
        k,
        -sim.w_ce / sim.w_ci
        - k
        * k_norm
        / w_norm
        * 3.0
        * np.sqrt(sim.T_plasma * constants.q_e / constants.m_e),
        c="pink",
        ls="-.",
        lw=1.25,
        label="$\omega = \Omega_{e} + 3v_{th,e} k$",
    )
    ax1.plot(
        k,
        -sim.w_ce / sim.w_ci
        + k
        * k_norm
        / w_norm
        * 3.0
        * np.sqrt(sim.T_plasma * constants.q_e / constants.m_e),
        c="pink",
        ls="-.",
        lw=1.25,
    )

else:
    ax1.plot(
        k,
        k * k_norm * sim.vA / w_norm,
        c="limegreen",
        ls="-.",
        lw=1.5,
        label="$\omega = v_Ak$",
    )

    x = [
        3.9732873563218387,
        3.6515862068965514,
        3.306275862068966,
        2.895655172413793,
        2.4318850574712645,
        2.0747586206896553,
        1.8520229885057473,
        1.6589195402298849,
        1.4594942528735633,
        1.2911724137931033,
        1.1551264367816092,
        1.0335402298850576,
        0.8961149425287356,
        0.7419770114942528,
        0.6141379310344828,
        0.4913103448275862,
    ]
    y = [
        1.1145945018655916,
        1.1193978642192393,
        1.1391259596002916,
        1.162971222713042,
        1.1986533430544237,
        1.230389844319595,
        1.2649997855641806,
        1.3265857528841618,
        1.3706737573444268,
        1.4368486511986962,
        1.4933310460179268,
        1.5485268259210019,
        1.6386327572157655,
        1.7062658146416778,
        1.7828194021529358,
        1.8533687867221342,
    ]
    ax1.plot(x, y, c="limegreen", ls=":", lw=2, label="Bernstein modes")

    x = [
        3.9669885057471266,
        3.6533333333333333,
        3.3213563218390805,
        2.9646896551724136,
        2.6106436781609195,
        2.2797011494252875,
        1.910919540229885,
        1.6811724137931034,
        1.4499540229885057,
        1.2577011494252872,
        1.081057471264368,
        0.8791494252873564,
        0.7153103448275862,
    ]
    y = [
        2.2274306300124374,
        2.2428271218424327,
        2.272505039241755,
        2.3084873697302397,
        2.3586224642964364,
        2.402667581592829,
        2.513873997512545,
        2.5859673199811297,
        2.6586610627439207,
        2.7352146502551786,
        2.8161427284813656,
        2.887850066475104,
        2.9455761890466183,
    ]
    ax1.plot(x, y, c="limegreen", ls=":", lw=2)

    x = [
        3.9764137931034487,
        3.702022988505747,
        3.459793103448276,
        3.166712643678161,
        2.8715862068965516,
        2.5285057471264367,
        2.2068505747126435,
        1.9037011494252871,
        1.6009885057471265,
        1.3447816091954023,
        1.1538850574712645,
        0.9490114942528736,
    ]
    y = [
        3.3231976669382854,
        3.34875841660591,
        3.378865205643951,
        3.424454260839731,
        3.474160483767209,
        3.522194107303684,
        3.6205343740618434,
        3.7040356821203417,
        3.785435519149119,
        3.868851052879873,
        3.9169704507440923,
        3.952481022429987,
    ]
    ax1.plot(x, y, c="limegreen", ls=":", lw=2)

    w_pi_SI = sim.w_pi * sim.w_pe_SI / sim.w_pe
    w_LH = 1.0 / np.sqrt(1.0 / (sim.w_ci * sim.w_ce) + 1.0 / w_pi_SI**2)
    ax1.axhline(w_LH / w_norm, ls="--", c="pink", label="$\omega_{LH}$")

# ax1.legend(loc='upper left')
fig.legend(loc=7, fontsize=18)

if sim.B_dir == "z":
    ax1.set_xlabel(r"$k l_i$")
    ax1.set_title("$B_{R/L} = B_x \pm iB_y$")
    fig.suptitle("Parallel modes")
    ax1.set_xlim(-4.5, 4.5)
    ax1.set_ylim(-12, 3)
    dir_str = "par"
else:
    ax1.set_xlabel(r"$k \rho_i$")
    ax1.set_title("$E_z(k, \omega)$")
    fig.suptitle(f"Perpendicular modes (ion Bernstein) - {sim.dim}D")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 8)
    dir_str = "perp"

ax1.set_ylabel(r"$\omega / \Omega_i$")

plt.savefig(
    f"spectrum_{dir_str}_{sim.dim}d_{sim.C_SI}_C_SI.png",
    bbox_inches="tight",
)
if not sim.test:
    plt.show()
