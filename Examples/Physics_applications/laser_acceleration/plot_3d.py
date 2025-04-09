#!/usr/bin/env python3

# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL
#
# This is a script plots the wakefield of an LWFA simulation.

import sys

import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(50)


def plot_lwfa():
    # this will be the name of the plot file
    fn = sys.argv[1]

    # Read the file
    ds = yt.load(fn)

    # plot the laser field and absolute density
    fields = ["Ey", "rho"]
    normal = "y"
    sl = yt.SlicePlot(ds, normal=normal, fields=fields)
    for field in fields:
        sl.set_log(field, False)

    sl.set_figure_size((4, 8))
    fig = sl.export_to_mpl_figure(nrows_ncols=(2, 1))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_lwfa()
