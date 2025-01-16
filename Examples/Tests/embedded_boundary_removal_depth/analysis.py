#!/usr/bin/env python

"""
This analysis script checks for any spurious charge build-up at the embedded boundary, when particles are removed in 3D.
It averages the divergence of the electric field (divE) over the last 20 time steps and compares the results with a specified tolerance.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from openpmd_viewer import OpenPMDTimeSeries

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
import checksumAPI

# Open plotfile specified in command line
filename = sys.argv[1]
test_name = os.path.split(os.getcwd())[1]
checksumAPI.evaluate_checksum(test_name, filename, output_format="openpmd")


def get_avg_divE(ts, start_avg_iter, end_avg_iter, ncell, test_dim):
    for iteration in tqdm.tqdm(ts.iterations[start_avg_iter:end_avg_iter]):
        if test_dim == "3d":
            avg_divE = np.zeros((ncell, ncell))
            divE = ts.get_field(
                "divE", iteration=iteration, slice_across="y", plot=False, cmap="RdBu"
            )[0]
        elif test_dim == "2d":
            avg_divE = np.zeros((ncell, ncell))
            divE = ts.get_field("divE", iteration=iteration, plot=False, cmap="RdBu")[0]
        elif test_dim == "rz":
            avg_divE = np.zeros((ncell, 2 * ncell))
            divE = ts.get_field("divE", iteration=iteration, plot=False, cmap="RdBu")[0]
        avg_divE += divE
    return avg_divE / (end_avg_iter - start_avg_iter)


def parse_dimension_in_test_name(test_name):
    test_name = test_name.lower()
    if "3d" in test_name:
        return "3d"
    elif "2d" in test_name:
        return "2d"
    elif "rz" in test_name:
        return "rz"
    return None


def plot(array):
    x = np.linspace(-7, 7, 400)
    y = np.sqrt(7**2 - x**2)

    fig, ax = plt.subplots()

    # Plot the boundary curves
    ax.plot(x, y, "k--")
    ax.plot(x, -y, "k--")
    cax = ax.imshow(array, cmap="RdBu", extent=[-10, 10, -10, 10], origin="lower")
    fig.colorbar(cax, ax=ax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Averaged divE")


ts = OpenPMDTimeSeries("./diags/diag1/")

ncell = 32
start_avg_iter = 30
end_avg_iter = 50
test_dim = parse_dimension_in_test_name(test_name)

divE_avg = get_avg_divE(ts, start_avg_iter, end_avg_iter, ncell, test_dim)
plot(divE_avg)
plt.savefig("AverageddivE.png")

if test_dim == "3d" or test_dim == "rz":
    tolerance = 1e-10
else:
    tolerance = 1e-9


def check_tolerance(array, tolerance):
    assert np.all(array <= tolerance), (
        f"Test did not pass: one or more elements exceed the tolerance of {tolerance}."
    )
    print("All elements of are within the tolerance.")


check_tolerance(divE_avg, tolerance)
