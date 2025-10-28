#!/usr/bin/env python3
#
# --- Analysis script for the hybrid-PIC example of magnetic reconnection.

import glob

import dill
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

plt.rcParams.update({"font.size": 20})

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

x_idx = 2
z_idx = 4
Ey_idx = 6
Bx_idx = 8

plane_data = np.loadtxt("diags/plane.dat", skiprows=1)

steps = np.unique(plane_data[:, 0])
num_steps = len(steps)
num_cells = plane_data.shape[0] // num_steps

plane_data = plane_data.reshape((num_steps, num_cells, plane_data.shape[1]))

times = plane_data[:, 0, 1]
dt = np.mean(np.diff(times))

plt.plot(
    times / sim.t_ci,
    np.mean(plane_data[:, :, Ey_idx], axis=1) / (sim.vA * sim.B0),
    "o-",
)

plt.grid()
plt.xlabel(r"$t/\tau_{c,i}$")
plt.ylabel("$<E_y>/v_AB_0$")
plt.title("Reconnection rate")
plt.tight_layout()
plt.savefig("diags/reconnection_rate.png")

if not sim.test:
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from scipy import interpolate

    # Animate the magnetic reconnection
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 9))

    for ax in axes.flatten():
        ax.set_aspect("equal")
        ax.set_ylabel("$z/l_i$")

    axes[2].set_xlabel("$x/l_i$")

    datafiles = sorted(glob.glob("diags/fields/*.npz"))
    num_steps = len(datafiles)

    data0 = np.load(datafiles[0])

    sX = axes[0].imshow(
        data0["Jy"].T,
        origin="lower",
        norm=colors.TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=1.6),
        extent=[0, sim.LX, -sim.LZ / 2, sim.LZ / 2],
        cmap=plt.cm.RdYlBu_r,
    )
    # axes[0].set_ylim(-5, 5)
    cb = plt.colorbar(sX, ax=axes[0], label="$J_y/J_0$")
    cb.ax.set_yscale("linear")
    cb.ax.set_yticks([-0.5, 0.0, 0.75, 1.5])

    sY = axes[1].imshow(
        data0["By"].T,
        origin="lower",
        extent=[0, sim.LX, -sim.LZ / 2, sim.LZ / 2],
        cmap=plt.cm.plasma,
    )
    # axes[1].set_ylim(-5, 5)
    cb = plt.colorbar(sY, ax=axes[1], label="$B_y/B_0$")
    cb.ax.set_yscale("linear")

    sZ = axes[2].imshow(
        data0["Bz"].T,
        origin="lower",
        extent=[0, sim.LX, -sim.LZ / 2, sim.LZ / 2],
        # norm=colors.TwoSlopeNorm(vmin=-0.02, vcenter=0., vmax=0.02),
        cmap=plt.cm.RdBu,
    )
    cb = plt.colorbar(sZ, ax=axes[2], label="$B_z/B_0$")
    cb.ax.set_yscale("linear")

    # plot field lines
    x_grid = np.linspace(0, sim.LX, data0["Bx"][:-1].shape[0])
    z_grid = np.linspace(-sim.LZ / 2.0, sim.LZ / 2.0, data0["Bx"].shape[1])

    n_lines = 10
    start_x = np.zeros(n_lines)
    start_x[: n_lines // 2] = sim.LX
    start_z = np.linspace(-sim.LZ / 2.0 * 0.9, sim.LZ / 2.0 * 0.9, n_lines)
    step_size = 1.0 / 100.0

    def get_field_lines(Bx, Bz):
        field_line_coords = []

        Bx_interp = interpolate.interp2d(x_grid, z_grid, Bx[:-1].T)
        Bz_interp = interpolate.interp2d(x_grid, z_grid, Bz[:, :-1].T)

        for kk, z in enumerate(start_z):
            path_x = [start_x[kk]]
            path_z = [z]

            ii = 0
            while ii < 10000:
                ii += 1
                Bx = Bx_interp(path_x[-1], path_z[-1])[0]
                Bz = Bz_interp(path_x[-1], path_z[-1])[0]

                # print(path_x[-1], path_z[-1], Bx, Bz)

                # normalize and scale
                B_mag = np.sqrt(Bx**2 + Bz**2)
                if B_mag == 0:
                    break

                dx = Bx / B_mag * step_size
                dz = Bz / B_mag * step_size

                x_new = path_x[-1] + dx
                z_new = path_z[-1] + dz

                if (
                    np.isnan(x_new)
                    or x_new <= 0
                    or x_new > sim.LX
                    or abs(z_new) > sim.LZ / 2
                ):
                    break

                path_x.append(x_new)
                path_z.append(z_new)

            field_line_coords.append([path_x, path_z])
        return field_line_coords

    field_lines = []
    for path in get_field_lines(data0["Bx"], data0["Bz"]):
        path_x = path[0]
        path_z = path[1]
        (ln,) = axes[2].plot(path_x, path_z, "--", color="k")
        # draws arrows on the field lines
        # if path_x[10] > path_x[0]:
        axes[2].arrow(
            path_x[50],
            path_z[50],
            path_x[250] - path_x[50],
            path_z[250] - path_z[50],
            shape="full",
            length_includes_head=True,
            lw=0,
            head_width=1.0,
            color="g",
        )

        field_lines.append(ln)

    def animate(i):
        data = np.load(datafiles[i])
        sX.set_array(data["Jy"].T)
        sY.set_array(data["By"].T)
        sZ.set_array(data["Bz"].T)
        sZ.set_clim(-np.max(abs(data["Bz"])), np.max(abs(data["Bz"])))

        for ii, path in enumerate(get_field_lines(data["Bx"], data["Bz"])):
            path_x = path[0]
            path_z = path[1]
            field_lines[ii].set_data(path_x, path_z)

    anim = FuncAnimation(fig, animate, frames=num_steps - 1, repeat=True)

    writervideo = FFMpegWriter(fps=14)
    anim.save("diags/mag_reconnection.mp4", writer=writervideo)
