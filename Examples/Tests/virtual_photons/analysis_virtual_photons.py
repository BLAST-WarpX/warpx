#!/usr/bin/env python3

# This test generates the population of virtual photons
# of one high-energy primary particle (electron, muon, or ion).
# The total number and spectrum of the virtual photons are
# compared to the theoretical prediction.
# Checks that the photons are in the same position of the primary particle.

import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import alpha, c, eV, m_e, pi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--charge_multiple",
    help="primary particle charge, in units of the elementary charge "
    "(number of virtual photons scales with the square of this value)",
    type=float,
    default=1.0,
)
args, unknown = parser.parse_known_args()

###########################
### ENERGY AND SPECTRUM ###
###########################

# useful constants
GeV = 1e9 * eV

# primary particle energy (same for all the virtual_photons tests, regardless
# of the primary species, so this does not need to be passed as an argument)
energy = 125 * GeV

# virtual photons min energy
hw_min = 1e-12 * m_e * c**2

# min fractional energy of the virtual photon wrt primary particle energy
ymin = hw_min / energy

#############
### WarpX ###
#############

series = OpenPMDTimeSeries("./diags/diag1/")
sampling_factor = 1e7
uz_virtual_photons, w_virtual_photons = series.get_particle(
    ["uz", "w"], species="virtual_photons", iteration=1
)
w_primary = series.get_particle(["w"], species="beam", iteration=1)

# fractional photon energy (photon energy / primary particle energy)
y_warpx = uz_virtual_photons * c / energy

# bins for the fractional photon energy
y = np.geomspace(ymin, 1, 401)

# number of virtual photons per primary particle obtained with WarpX
N_warpx = np.sum(w_virtual_photons) / np.sum(w_primary)

# spectrum of the virtual photons per primary particle
H, b = np.histogram(y_warpx, bins=y, weights=w_virtual_photons)
db = np.diff(b)
b = 0.5 * (b[1:] + b[:-1])
dN_dy_warpx = H / db / np.sum(w_primary)

##############
### Theory ###
##############

y = b
# number of virtual photons scales with the square of the primary particle's charge
charge_sq = args.charge_multiple**2
# spectrum of virtual photons for one primary particle
dN_dy_theory = charge_sq * alpha / pi / y * (-2 * log(y))
# dN_dy[dN_dy < 0] = 0.0

# number of virtual photons for one primary particle from theory
N_theory = charge_sq * alpha / pi * log(ymin) ** 2

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

print("Number of virtual photons per primary particle:")
print(f"From simulation : {N_warpx}")
print(f"From theory     : {N_theory}")
print(f"Relative error  : {number_rel_error:.4%}")

print("Spectrum of virtual photons per primary particle:")
print(f"Max relative error: {spectrum_rel_error.max()}")

assert number_rel_error < 0.02
assert (spectrum_rel_error < 0.04).all()

################
### Position ###
################

x, y, z = series.get_particle(["x", "y", "z"], species="virtual_photons", iteration=1)
x_primary, y_primary, z_primary = series.get_particle(
    ["x", "y", "z"], species="beam", iteration=1
)

assert np.unique(x) == x_primary
assert np.unique(y) == y_primary
assert np.unique(z) == z_primary
