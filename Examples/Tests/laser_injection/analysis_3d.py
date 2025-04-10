#!/usr/bin/env python3

# Copyright 2019 Andrew Myers, Jean-Luc Vay, Maxence Thevenet
# Remi Lehe, Weiqun Zhang
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# you can save an image to be displayed on the website
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)
plt.plot(t, s)
plt.savefig("laser_analysis.png")
