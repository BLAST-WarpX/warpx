#!/usr/bin/env python3
#
# --- Analysis script for the hybrid-PIC example of ion beam R instability.

import dill
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pywarpx import picmi

constants = picmi.constants

matplotlib.rcParams.update({"font.size": 20})

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

if sim.resonant:
    resonant_str = "resonant"
else:
    resonant_str = "non resonant"

data = np.loadtxt("diags/field_data.txt", skiprows=1)
if sim.dim == 1:
    field_idx_dict = {"z": 4, "By": 8}
else:
    field_idx_dict = {"z": 2, "By": 3}

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

data = np.zeros((num_steps, resolution))
for i in range(num_steps):
    data[i, :] = sim_data[i, idx, field_idx_dict["By"]]

print(f"Data file contains {num_steps} time snapshots.")
print(f"Spatial resolution is {resolution}")

# Create the stack time plot
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

max_val = np.max(np.abs(data[:, :] / sim.B0))

extent = [0, sim.Lz / sim.l_i, 0, num_steps * dt * sim.w_ci]  # num_steps*dt/sim.t_ci]
im = ax1.imshow(
    data[:, :] / sim.B0,
    extent=extent,
    origin="lower",
    cmap="seismic",
    vmin=-max_val,
    vmax=max_val,
    aspect="equal",
)

# Colorbar
fig.subplots_adjust(right=0.825)
cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="$B_y/B_0$")

ax1.set_xlabel("$x/l_i$")
ax1.set_ylabel("$t \Omega_i$ (rad)")

ax1.set_title(f"Ion beam R instability - {resonant_str} case")
plt.savefig(
    f"diags/ion_beam_R_instability_{resonant_str}_eta_{sim.eta}_substeps_{sim.substeps}.png"
)
plt.close()

if sim.resonant:
    # Plot the 4th, 5th and 6th Fourier modes
    field_kt = np.fft.fft(data[:, :], axis=1)
    k = 2 * np.pi * np.fft.fftfreq(resolution, dz) * sim.l_i

    t_grid = np.arange(num_steps) * dt * sim.w_ci
    plt.plot(
        t_grid, np.abs(field_kt[:, 4] / sim.B0), "r", label=f"m = 4, $kl_i={k[4]:.2f}$"
    )
    plt.plot(
        t_grid, np.abs(field_kt[:, 5] / sim.B0), "b", label=f"m = 5, $kl_i={k[5]:.2f}$"
    )
    plt.plot(
        t_grid, np.abs(field_kt[:, 6] / sim.B0), "k", label=f"m = 6, $kl_i={k[6]:.2f}$"
    )

    # The theoretical growth rates for the 4th, 5th and 6th Fourier modes of
    # the By-field was obtained from Fig. 12a of Munoz et al.
    # Note the rates here are gamma / w_ci
    gamma4 = 0.1915611861780133
    gamma5 = 0.20087036355662818
    gamma6 = 0.17123024228396777

    # Draw the line of best fit with the theoretical growth rate (slope) in the
    # window t*w_ci between 10 and 40
    idx = np.where((t_grid > 10) & (t_grid < 40))
    t_points = t_grid[idx]

    A4 = np.exp(np.mean(np.log(np.abs(field_kt[idx, 4] / sim.B0)) - t_points * gamma4))
    plt.plot(t_points, A4 * np.exp(t_points * gamma4), "r--", lw=3)
    A5 = np.exp(np.mean(np.log(np.abs(field_kt[idx, 5] / sim.B0)) - t_points * gamma5))
    plt.plot(t_points, A5 * np.exp(t_points * gamma5), "b--", lw=3)
    A6 = np.exp(np.mean(np.log(np.abs(field_kt[idx, 6] / sim.B0)) - t_points * gamma6))
    plt.plot(t_points, A6 * np.exp(t_points * gamma6), "k--", lw=3)

    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.ylabel("$|B_y/B_0|$")
    plt.xlabel("$t\Omega_i$ (rad)")
    plt.tight_layout()
    plt.savefig(
        f"diags/ion_beam_R_instability_{resonant_str}_eta_{sim.eta}_substeps_{sim.substeps}_low_modes.png"
    )
    plt.close()

    # check if the growth rate matches expectation
    m4_rms_error = np.sqrt(
        np.mean(
            (np.abs(field_kt[idx, 4] / sim.B0) - A4 * np.exp(t_points * gamma4)) ** 2
        )
    )
    m5_rms_error = np.sqrt(
        np.mean(
            (np.abs(field_kt[idx, 5] / sim.B0) - A5 * np.exp(t_points * gamma5)) ** 2
        )
    )
    m6_rms_error = np.sqrt(
        np.mean(
            (np.abs(field_kt[idx, 6] / sim.B0) - A6 * np.exp(t_points * gamma6)) ** 2
        )
    )
    print("Growth rate RMS errors:")
    print(f"    m = 4: {m4_rms_error:.3e}")
    print(f"    m = 5: {m5_rms_error:.3e}")
    print(f"    m = 6: {m6_rms_error:.3e}")

if not sim.test:
    with h5py.File("diags/Python_hybrid_PIC_plt/openpmd_004000.h5", "r") as data:
        timestep = str(np.squeeze([key for key in data["data"].keys()]))

        z = np.array(data["data"][timestep]["particles"]["ions"]["position"]["z"])
        vy = np.array(data["data"][timestep]["particles"]["ions"]["momentum"]["y"])
        w = np.array(data["data"][timestep]["particles"]["ions"]["weighting"])

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    im = ax1.hist2d(
        z / sim.l_i,
        vy / sim.M / sim.vA,
        weights=w,
        density=True,
        range=[[0, 250], [-10, 10]],
        bins=250,
        cmin=1e-5,
    )

    # Colorbar
    fig.subplots_adjust(bottom=0.15, right=0.815)
    cbar_ax = fig.add_axes([0.83, 0.2, 0.03, 0.6])
    fig.colorbar(
        im[3], cax=cbar_ax, orientation="vertical", format="%.0e", label="$f(z, v_y)$"
    )

    ax1.set_xlabel("$x/l_i$")
    ax1.set_ylabel("$v_{y}/v_A$")

    ax1.set_title(f"Ion beam R instability - {resonant_str} case")
    plt.savefig(
        f"diags/ion_beam_R_instability_{resonant_str}_eta_{sim.eta}_substeps_{sim.substeps}_core_phase_space.png"
    )
    plt.close()

    with h5py.File("diags/Python_hybrid_PIC_plt/openpmd_004000.h5", "r") as data:
        timestep = str(np.squeeze([key for key in data["data"].keys()]))

        z = np.array(data["data"][timestep]["particles"]["beam_ions"]["position"]["z"])
        vy = np.array(data["data"][timestep]["particles"]["beam_ions"]["momentum"]["y"])
        w = np.array(data["data"][timestep]["particles"]["beam_ions"]["weighting"])

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    im = ax1.hist2d(
        z / sim.l_i,
        vy / sim.M / sim.vA,
        weights=w,
        density=True,
        range=[[0, 250], [-10, 10]],
        bins=250,
        cmin=1e-5,
    )

    # Colorbar
    fig.subplots_adjust(bottom=0.15, right=0.815)
    cbar_ax = fig.add_axes([0.83, 0.2, 0.03, 0.6])
    fig.colorbar(
        im[3], cax=cbar_ax, orientation="vertical", format="%.0e", label="$f(z, v_y)$"
    )

    ax1.set_xlabel("$x/l_i$")
    ax1.set_ylabel("$v_{y}/v_A$")

    ax1.set_title(f"Ion beam R instability - {resonant_str} case")
    plt.savefig(
        f"diags/ion_beam_R_instability_{resonant_str}_eta_{sim.eta}_substeps_{sim.substeps}_beam_phase_space.png"
    )
    plt.show()

if sim.test:
    # physics based check - these error tolerances are not set from theory
    # but from the errors that were present when the test was created. If these
    # assert's fail, the full benchmark should be rerun (same as the test but
    # without the `--test` argument) and the growth rates (up to saturation)
    # compared to the theoretical ones to determine if the physics test passes.
    # At creation, the full test (3d) had the following errors (ran on 1 V100):
    # m4_rms_error = 3.329; m5_rms_error = 1.052; m6_rms_error = 2.583
    assert np.isclose(m4_rms_error, 1.515, atol=0.01)
    assert np.isclose(m5_rms_error, 0.718, atol=0.01)
    assert np.isclose(m6_rms_error, 0.357, atol=0.01)
