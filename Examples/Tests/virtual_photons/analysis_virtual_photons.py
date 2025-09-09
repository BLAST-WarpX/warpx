#!/usr/bin/env python3

# This test generates the population of virtual photons
# of one high-energy electron.
# The total number and spectrum of the virtual photons are
# compared to the theoretical prediction.
# Checks that the photons are in the same position of the electron.

import numpy as np
from numpy import log
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import alpha, c, eV, m_e, pi

###########################
### ENERGY AND SPECTRUM ###
###########################

# useful constants
GeV = 1e9 * eV
gamma_em = 0.5772156649015  # Euler-Mascheroni

# electron energy
energy = 125 * GeV
gamma = energy / (m_e * c**2)
gamma0 = 125 * GeV / (m_e * c**2)

# virtual photons min energy
hw_min = 1e-12 * m_e * c**2

# min fractional energy of the virtual photon wrt electron energy
ymin = hw_min / energy

#############
### WarpX ###
#############

series = OpenPMDTimeSeries("./diags/diag1/")
sampling_factor = 10000000
uz, w_wx = series.get_particle(["uz", "w"], species="virtual_photons", iteration=1)
w_primary_wx = series.get_particle(["w"], species="beam", iteration=1)

# fractional photon energy (photon energy / electron energy)
y_wx = uz * c / energy

# bins for the fractional photon energy
y = np.geomspace(ymin, 1, 401)

# number of virtual photons per electron obtained with WarpX
N_wx = np.sum(w_wx) / np.sum(w_primary_wx)

# spectrum of the virtual photons per electron
H, b = np.histogram(y_wx, bins=y, weights=w_wx)
db = np.diff(b)
b = 0.5 * (b[1:] + b[:-1])
dN_dy_wx = H / db / np.sum(w_primary_wx)

##############
### Theory ###
##############

y = b
# spectrum of virtual photons for one electron
A = 0  # log(4) - 2 * gamma_em - 1 (if not neglecting some terms)
dN_dy = alpha / pi / y * (-2 * log(y) + A)
# dN_dy[dN_dy < 0] = 0.0

# number of virtual photons for one electron from theory
N = alpha / pi * (-A * log(ymin) + log(ymin) ** 2)

print("Number of virtual photons per electron:")
print(f"From simulation : {N_wx}")
print(f"From theory     : {N}")
print(f"Relative error  : {abs(N_wx - N) / N:.2%}")

print("Spectrum of virtual photons per electron:")
print(f"Max relative error: {(np.abs(dN_dy_wx - dN_dy) / dN_dy).max()}")
assert (np.abs(dN_dy_wx - dN_dy) < 0.04 * dN_dy).all()

assert abs(N - N_wx) < 0.02 * N


################
### POSITION ###
################

x, y, z = series.get_particle(["x", "y", "z"], species="virtual_photons", iteration=1)
x_e, y_e, z_e = series.get_particle(["x", "y", "z"], species="beam", iteration=1)

assert np.unique(x) == x_e
assert np.unique(y) == y_e
assert np.unique(z) == z_e
