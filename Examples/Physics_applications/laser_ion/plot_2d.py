#!/usr/bin/env python3

# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Marco Garten
# License: BSD-3-Clause-LBNL
#
# This script plots the densities and fields of a 2D laser-ion acceleration simulation.


import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sc
from matplotlib.colors import TwoSlopeNorm
from openpmd_viewer import OpenPMDTimeSeries

plt.rcParams.update({"font.size": 16})


def create_analysis_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def visualize_density_iteration(ts, iteration, out_dir):
    """
    Visualize densities and fields of a single iteration.

    :param ts: OpenPMDTimeSeries
    :param iteration: Output iteration (simulation timestep)
    :param out_dir: Directory for PNG output
    :return:
    """
    # Physics parameters
    lambda_L = 800e-9  # Laser wavelength in meters
    omega_L = 2 * np.pi * sc.c / lambda_L  # Laser frequency in seconds
    n_c = (
        sc.m_e * sc.epsilon_0 * omega_L**2 / sc.elementary_charge**2
    )  # Critical plasma density in meters^(-3)
    micron = 1e-6

    # Simulation parameters
    n_e0 = 30
    n_max = 2 * n_e0
    nr = 1  # Number to decrease resolution

    # Data fetching
    it = iteration
    ii = np.where(ts.iterations == it)[0][0]

    time = ts.t[ii]
    rho_e, rho_e_info = ts.get_field(field="rho_electrons", iteration=it)
    rho_d, rho_d_info = ts.get_field(field="rho_hydrogen", iteration=it)

    # Rescale to critical density
    rho_e = rho_e / (sc.elementary_charge * n_c)
    rho_d = rho_d / (sc.elementary_charge * n_c)

    # Axes setup
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    xax, zax = rho_e_info.x, rho_e_info.z

    # Plotting
    # Electron density
    im0 = axs[0].pcolormesh(
        zax[::nr] / micron,
        xax[::nr] / micron,
        -rho_e.T[::nr, ::nr],
        vmin=0,
        vmax=n_max,
        cmap="Reds",
        rasterized=True,
    )
    plt.colorbar(im0, ax=axs[0], label=r"$n_\mathrm{\,e}\ (n_\mathrm{c})$")

    # Hydrogen density
    im1 = axs[1].pcolormesh(
        zax[::nr] / micron,
        xax[::nr] / micron,
        rho_d.T[::nr, ::nr],
        vmin=0,
        vmax=n_max,
        cmap="Blues",
        rasterized=True,
    )
    plt.colorbar(im1, ax=axs[1], label=r"$n_\mathrm{\,H}\ (n_\mathrm{c})$")

    # Masked electron density
    divnorm = TwoSlopeNorm(vmin=-7.0, vcenter=0.0, vmax=2)
    masked_data = np.ma.masked_where(rho_e.T == 0, rho_e.T)
    my_cmap = plt.cm.PiYG_r.copy()
    my_cmap.set_bad(color="black")
    im2 = axs[2].pcolormesh(
        zax[::nr] / micron,
        xax[::nr] / micron,
        np.log(-masked_data[::nr, ::nr]),
        norm=divnorm,
        cmap=my_cmap,
        rasterized=True,
    )
    plt.colorbar(
        im2,
        ax=axs[2],
        ticks=[-6, -3, 0, 1, 2],
        extend="both",
        label=r"$\log n_\mathrm{\,e}\ (n_\mathrm{c})$",
    )

    # Axis labels and title
    for ax in axs:
        ax.set_aspect(1.0)
        ax.set_ylabel(r"$x$ ($\mu$m)")
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    axs[2].set_xlabel(r"$z$ ($\mu$m)")
    fig.suptitle(f"Iteration: {it}, Time: {time / 1e-15:.1f} fs")

    plt.tight_layout()

    plt.savefig(f"{out_dir}/densities_{it:06d}.png")


def visualize_field_iteration(ts, iteration, out_dir):
    # Additional parameters
    nr = 1  # Number to decrease resolution
    micron = 1e-6

    # Data fetching
    it = iteration
    ii = np.where(ts.iterations == it)[0][0]
    time = ts.t[ii]

    Ex, Ex_info = ts.get_field(field="E", coord="x", iteration=it)
    Exmax = np.max(np.abs([np.min(Ex), np.max(Ex)]))
    By, By_info = ts.get_field(field="B", coord="y", iteration=it)
    Bymax = np.max(np.abs([np.min(By), np.max(By)]))
    Ez, Ez_info = ts.get_field(field="E", coord="z", iteration=it)
    Ezmax = np.max(np.abs([np.min(Ez), np.max(Ez)]))

    # Axes setup
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    xax, zax = Ex_info.x, Ex_info.z

    # Plotting
    im0 = axs[0].pcolormesh(
        zax[::nr] / micron,
        xax[::nr] / micron,
        Ex.T[::nr, ::nr],
        vmin=-Exmax,
        vmax=Exmax,
        cmap="RdBu",
        rasterized=True,
    )

    plt.colorbar(im0, ax=axs[00], label=r"$E_x$ (V/m)")

    im1 = axs[1].pcolormesh(
        zax[::nr] / micron,
        xax[::nr] / micron,
        By.T[::nr, ::nr],
        vmin=-Bymax,
        vmax=Bymax,
        cmap="RdBu",
        rasterized=True,
    )
    plt.colorbar(im1, ax=axs[1], label=r"$B_y$ (T)")

    im2 = axs[2].pcolormesh(
        zax[::nr] / micron,
        xax[::nr] / micron,
        Ez.T[::nr, ::nr],
        vmin=-Ezmax,
        vmax=Ezmax,
        cmap="RdBu",
        rasterized=True,
    )
    plt.colorbar(im2, ax=axs[2], label=r"$E_z$ (V/m)")

    # Axis labels and title
    for ax in axs:
        ax.set_aspect(1.0)
        ax.set_ylabel(r"$x$ ($\mu$m)")
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    axs[2].set_xlabel(r"$z$ ($\mu$m)")
    fig.suptitle(f"Iteration: {it}, Time: {time / 1e-15:.1f} fs")

    plt.tight_layout()

    plt.savefig(f"{out_dir}/fields_{it:06d}.png")


def visualize_particle_histogram_iteration(
    diag_name="histuH", species="hydrogen", iteration=1000, out_dir="./analysis"
):
    it = iteration

    if species == "hydrogen":
        # proton rest energy in eV
        mc2 = sc.m_p / sc.electron_volt * sc.c**2
    elif species == "electron":
        mc2 = sc.m_e / sc.electron_volt * sc.c**2
    else:
        raise NotImplementedError(
            "The only implemented presets for this analysis script are `electron` or `hydrogen`."
        )

    fs = 1.0e-15
    MeV = 1.0e6

    df = pd.read_csv(f"./diags/reducedfiles/{diag_name}.txt", delimiter=r"\s+")
    # the columns look like this:
    #     #[0]step() [1]time(s) [2]bin1=0.000220() [3]bin2=0.000660() [4]bin3=0.001100()

    # matches words, strings surrounded by " ' ", dots, minus signs and e for scientific notation in numbers
    nested_list = [re.findall(r"[\w'\.]+", col) for col in df.columns]

    index = pd.MultiIndex.from_tuples(
        nested_list, names=("column#", "name", "bin value")
    )

    df.columns = index

    steps = df.values[:, 0].astype(int)
    ii = np.where(steps == it)[0][0]
    time = df.values[:, 1]
    data = df.values[:, 2:]
    edge_vals = np.array([float(row[2]) for row in df.columns[2:]])
    edges_MeV = (np.sqrt(edge_vals**2 + 1) - 1) * mc2 / MeV

    time_fs = time / fs

    fig, ax = plt.subplots(1, 1)

    ax.plot(edges_MeV, data[ii, :])
    ax.set_yscale("log")
    ax.set_ylabel(r"d$N$/d$\mathcal{E}$ (arb. u.)")
    ax.set_xlabel(r"$\mathcal{E}$ (MeV)")

    fig.suptitle(f"{species} - Iteration: {it}, Time: {time_fs[ii]:.1f} fs")

    plt.tight_layout()
    plt.savefig(f"./{out_dir}/{diag_name}_{it:06d}.png")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Visualize Laser-Ion Accelerator Densities and Fields"
    )
    parser.add_argument(
        "-d",
        "--diag_dir",
        type=str,
        default="./diags/diagInst",
        help="Directory containing density and field diagnostics",
    )
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=None,
        help="Specific iteration to visualize",
    )
    parser.add_argument(
        "-hn",
        "--histogram_name",
        type=str,
        default="histuH",
        help="Name of histogram diagnostic to visualize",
    )
    parser.add_argument(
        "-hs",
        "--histogram_species",
        type=str,
        default="hydrogen",
        help="Particle species in the visualized histogram diagnostic",
    )
    args = parser.parse_args()

    # Create analysis directory
    analysis_dir = "analysis"
    create_analysis_dir(analysis_dir)

    # Loading the time series
    ts = OpenPMDTimeSeries(args.diag_dir)

    if args.iteration is not None:
        visualize_density_iteration(ts, args.iteration, analysis_dir)
        visualize_field_iteration(ts, args.iteration, analysis_dir)
        visualize_particle_histogram_iteration(
            args.histogram_name, args.histogram_species, args.iteration, analysis_dir
        )
    else:
        for it in ts.iterations:
            visualize_density_iteration(ts, it, analysis_dir)
            visualize_field_iteration(ts, it, analysis_dir)
            visualize_particle_histogram_iteration(
                args.histogram_name, args.histogram_species, it, analysis_dir
            )
