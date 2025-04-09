#!/usr/bin/env python3

# This script displays a 2D histogram.

import argparse

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

parser = argparse.ArgumentParser(
    description="Process a 2D histogram name and an integer."
)
parser.add_argument("hist2D", help="Folder name of the reduced diagnostic.")
parser.add_argument(
    "iter",
    help="Iteration number of the simulation that is plotted. Enter a number from the list of iterations or 'All' if you want all plots.",
)
args = parser.parse_args()

path = "diags/reducedfiles/" + args.hist2D

ts = OpenPMDTimeSeries(path)

it = ts.iterations
data, info = ts.get_field(field="data", iteration=0, plot=True)
print("The available iterations of the simulation are:", it)
print("The axes of the histogram are (0: ordinate ; 1: abscissa):", info.axes)
print("The data shape is:", data.shape)

# Add the simulation time to the title once this information
# is available in the "info" FieldMetaInformation object.
if args.iter == "All":
    for it_idx, i in enumerate(it):
        plt.figure()
        data, info = ts.get_field(field="data", iteration=i, plot=False)
        abscissa_name = info.axes[1]  # This might be 'z' or something else
        abscissa_values = getattr(info, abscissa_name, None)
        ordinate_name = info.axes[0]  # This might be 'z' or something else
        ordinate_values = getattr(info, ordinate_name, None)

        plt.pcolormesh(
            abscissa_values / 1e-6,
            ordinate_values,
            data,
            norm=colors.LogNorm(),
            rasterized=True,
        )
        plt.title(args.hist2D + f" Time: {ts.t[it_idx]:.2e} s  (Iteration: {i:d})")
        plt.xlabel(info.axes[1] + r" ($\mu$m)")
        plt.ylabel(info.axes[0] + r" ($m_\mathrm{species} c$)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("Histogram_2D_" + args.hist2D + "_iteration_" + str(i) + ".png")
else:
    i = int(args.iter)
    it_idx = np.where(i == it)[0][0]
    plt.figure()
    data, info = ts.get_field(field="data", iteration=i, plot=False)
    abscissa_name = info.axes[1]  # This might be 'z' or something else
    abscissa_values = getattr(info, abscissa_name, None)
    ordinate_name = info.axes[0]  # This might be 'z' or something else
    ordinate_values = getattr(info, ordinate_name, None)

    plt.pcolormesh(
        abscissa_values / 1e-6,
        ordinate_values,
        data,
        norm=colors.LogNorm(),
        rasterized=True,
    )
    plt.title(args.hist2D + f" Time: {ts.t[it_idx]:.2e} s  (Iteration: {i:d})")
    plt.xlabel(info.axes[1] + r" ($\mu$m)")
    plt.ylabel(info.axes[0] + r" ($m_\mathrm{species} c$)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("Histogram_2D_" + args.hist2D + "_iteration_" + str(i) + ".png")
