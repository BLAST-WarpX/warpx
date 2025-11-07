#!/usr/bin/env python3

# This test generates the population of virtual photons
# of one high-energy electron.
# The total number and spectrum of the virtual photons are
# compared to the theoretical prediction.
# Checks that the photons are in the same position of the electron.

import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import alpha, c, eV, m_e, pi

###########################
### ENERGY AND SPECTRUM ###
###########################

# useful constants
GeV = 1e9 * eV

# electron energy
energy = 125 * GeV

# virtual photons min energy
hw_min = 1e-12 * m_e * c**2

# min fractional energy of the virtual photon wrt electron energy
ymin = hw_min / energy

#############
### WarpX ###
#############

series = OpenPMDTimeSeries("./diags/diag1/")
sampling_factor = 1e7
uz_virtual_photons, w_virtual_photons = series.get_particle(
    ["uz", "w"], species="virtual_photons", iteration=1
)
w_electrons = series.get_particle(["w"], species="beam", iteration=1)

# fractional photon energy (photon energy / electron energy)
y_warpx = uz_virtual_photons * c / energy

# bins for the fractional photon energy
y = np.geomspace(ymin, 1, 401)

# number of virtual photons per electron obtained with WarpX
N_warpx = np.sum(w_virtual_photons) / np.sum(w_electrons)

# spectrum of the virtual photons per electron
H, b = np.histogram(y_warpx, bins=y, weights=w_virtual_photons)
db = np.diff(b)
b = 0.5 * (b[1:] + b[:-1])
dN_dy_warpx = H / db / np.sum(w_electrons)

##############
### Theory ###
##############

y = b
# spectrum of virtual photons for one electron
dN_dy_theory = alpha / pi / y * (-2 * log(y))
# dN_dy[dN_dy < 0] = 0.0

# number of virtual photons for one electron from theory
N_theory = alpha / pi * log(ymin) ** 2

############
### Plot ###
############

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 4), dpi=200)
ax.plot(y, dN_dy_theory, color="black", lw=6, label="theory")
ax.plot(y, dN_dy_warpx, color="dodgerblue", lw=4, label="WarpX")
ax.legend()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Fractional photon energy")
ax.set_ylabel("dN/dy")
ax.set_title("Virtual photons spectrum")
fig.savefig("spectrum_virtual_photons.png")

#############
### Error ###
#############

number_rel_error = np.abs(N_warpx - N_theory) / N_theory
spectrum_rel_error = np.abs(dN_dy_warpx - dN_dy_theory) / dN_dy_theory

print("Number of virtual photons per electron:")
print(f"From simulation : {N_warpx}")
print(f"From theory     : {N_theory}")
print(f"Relative error  : {number_rel_error:.4%}")

print("Spectrum of virtual photons per electron:")
print(f"Max relative error: {spectrum_rel_error.max()}")

assert number_rel_error < 0.02
assert (spectrum_rel_error < 0.04).all()

################
### Position ###
################

x, y, z = series.get_particle(["x", "y", "z"], species="virtual_photons", iteration=1)
x_e, y_e, z_e = series.get_particle(["x", "y", "z"], species="beam", iteration=1)

assert np.unique(x) == x_e
assert np.unique(y) == y_e
assert np.unique(z) == z_e
