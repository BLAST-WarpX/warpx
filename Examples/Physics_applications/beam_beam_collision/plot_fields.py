#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from openpmd_viewer import OpenPMDTimeSeries

plt.rcParams.update({"font.size": 16})

series = OpenPMDTimeSeries("./diags/diag1")
steps = series.iterations


for slice_axis in ["x", "y"]:  # slice the fields along x and y
    for n in steps:  # loop through the available timesteps
        fig, ax = plt.subplots(
            ncols=2, nrows=2, figsize=(10, 6), dpi=300, sharex=True, sharey=True
        )

        # get E field
        Ex, info = series.get_field(
            field="E", coord="x", iteration=n, plot=False, slice_across=slice_axis
        )
        Ey, info = series.get_field(
            field="E", coord="y", iteration=n, plot=False, slice_across=slice_axis
        )
        Ez, info = series.get_field(
            field="E", coord="z", iteration=n, plot=False, slice_across=slice_axis
        )
        # get B field
        Bx, info = series.get_field(
            field="B", coord="x", iteration=n, plot=False, slice_across=slice_axis
        )
        By, info = series.get_field(
            field="B", coord="y", iteration=n, plot=False, slice_across=slice_axis
        )
        Bz, info = series.get_field(
            field="B", coord="z", iteration=n, plot=False, slice_across=slice_axis
        )
        # get charge densities
        rho_beam1, info = series.get_field(
            field="rho_beam1", iteration=n, plot=False, slice_across=slice_axis
        )
        rho_beam2, info = series.get_field(
            field="rho_beam2", iteration=n, plot=False, slice_across=slice_axis
        )
        rho_ele1, info = series.get_field(
            field="rho_ele1", iteration=n, plot=False, slice_across=slice_axis
        )
        rho_pos1, info = series.get_field(
            field="rho_pos1", iteration=n, plot=False, slice_across=slice_axis
        )
        rho_ele2, info = series.get_field(
            field="rho_ele2", iteration=n, plot=False, slice_across=slice_axis
        )
        rho_pos2, info = series.get_field(
            field="rho_pos2", iteration=n, plot=False, slice_across=slice_axis
        )

        xmin = info.z.min()
        xmax = info.z.max()
        xlabel = "z [m]"

        if slice_axis == "x":
            ymin = info.y.min()
            ymax = info.y.max()
            ylabel = "y [m]"
        elif slice_axis == "y":
            ymin = info.x.min()
            ymax = info.x.max()
            ylabel = "x [m]"

        # plot E magnitude
        Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        im = ax[0, 0].imshow(
            np.transpose(Emag),
            cmap="seismic",
            extent=[xmin, xmax, ymin, ymax],
            vmin=0,
            vmax=np.max(np.abs(Emag)),
        )
        ax[0, 0].set_title("E [V/m]")
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        # plot B magnitude
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        im = ax[1, 0].imshow(
            np.transpose(Bmag),
            cmap="seismic",
            extent=[xmin, xmax, ymin, ymax],
            vmin=0,
            vmax=np.max(np.abs(Bmag)),
        )
        ax[1, 0].set_title("B [T]")
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        # plot beam densities
        rho_beams = rho_beam1 + rho_beam2
        im = ax[0, 1].imshow(
            np.transpose(rho_beams),
            cmap="seismic",
            extent=[xmin, xmax, ymin, ymax],
            vmin=-np.max(np.abs(rho_beams)),
            vmax=np.max(np.abs(rho_beams)),
        )
        ax[0, 1].set_title(r"$\rho$ beams [C/m$^3$]")
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        # plot secondary densities
        rho2 = rho_ele1 + rho_pos1 + rho_ele2 + rho_pos2
        im = ax[1, 1].imshow(
            np.transpose(rho2),
            cmap="seismic",
            extent=[xmin, xmax, ymin, ymax],
            vmin=-np.max(np.abs(rho2)),
            vmax=np.max(np.abs(rho2)),
        )
        ax[1, 1].set_title(r"$\rho$ secondaries [C/m$^3$]")
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        for a in ax[-1, :].reshape(-1):
            a.set_xlabel(xlabel)
        for a in ax[:, 0].reshape(-1):
            a.set_ylabel(ylabel)

        fig.suptitle(f"Iteration = {n}, time [s] = {series.current_t}", fontsize=20)
        plt.tight_layout()

        image_file_name = "FIELDS_" + slice_axis + f"_{n:03d}.png"
        plt.savefig(image_file_name, dpi=100, bbox_inches="tight")
        plt.close()
