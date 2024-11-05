#!/usr/bin/env python3
#
# --- Analysis script for the hybrid-PIC example of ion Landau damping.

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

# theoretical damping rates were taken from Fig. 14b of Munoz et al.
theoretical_damping_rate = np.array(
    [
        [0.09456706, 0.05113443],
        [0.09864177, 0.05847507],
        [0.10339559, 0.0659153],
        [0.10747029, 0.07359366],
        [0.11290323, 0.08256106],
        [0.11833616, 0.09262114],
        [0.12580645, 0.10541121],
        [0.13327674, 0.11825558],
        [0.14006791, 0.13203098],
        [0.14889643, 0.14600538],
        [0.15772496, 0.16379615],
        [0.16791171, 0.18026693],
        [0.17606112, 0.19650209],
        [0.18828523, 0.21522808],
        [0.19983022, 0.23349062],
        [0.21273345, 0.25209216],
        [0.22835314, 0.27877403],
        [0.24465195, 0.30098317],
        [0.25959253, 0.32186286],
        [0.27657046, 0.34254601],
        [0.29626486, 0.36983567],
        [0.3139219, 0.38984826],
        [0.33157895, 0.40897973],
        [0.35195246, 0.43526107],
        [0.37368421, 0.45662113],
        [0.39745331, 0.47902942],
        [0.44974533, 0.52973074],
        [0.50747029, 0.57743925],
        [0.57334465, 0.63246726],
        [0.64193548, 0.67634255],
    ]
)

expected_gamma = np.interp(
    sim.T_ratio, theoretical_damping_rate[:, 0], theoretical_damping_rate[:, 1]
)

data = np.loadtxt("diags/field_data.txt", skiprows=1)
field_idx_dict = {"z": 2, "Ez": 3}

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
    data[i, :] = sim_data[i, idx, field_idx_dict["Ez"]]

print(f"Data file contains {num_steps} time snapshots.")
print(f"Spatial resolution is {resolution}")

field_kt = np.fft.fft(data[:, :], axis=1)

t_norm = 2.0 * np.pi * sim.m / sim.Lz * sim.v_ti

# Plot the 4th Fourier mode
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

t_points = np.arange(num_steps) * dt * t_norm
ax1.plot(
    t_points,
    np.abs(field_kt[:, sim.m] / field_kt[0, sim.m]),
    "r",
    label=f"$T_i/T_e$ = {sim.T_ratio:.2f}",
)

# Plot a line showing the expected damping rate
t_points = t_points[np.where(t_points < 8)]
ax1.plot(t_points, np.exp(-t_points * expected_gamma), "k--", lw=2)

ax1.grid()
ax1.legend()
ax1.set_yscale("log")
ax1.set_ylabel("$|E_z|/E_0$")
ax1.set_xlabel("t $(k_mv_{th,i})$")
ax1.set_xlim(0, 18)

ax1.set_title(f"Ion Landau damping - {sim.dim}d")
plt.tight_layout()
plt.savefig(f"diags/ion_Landau_damping_T_ratio_{sim.T_ratio}.png")
