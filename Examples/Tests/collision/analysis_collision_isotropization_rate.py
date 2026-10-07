#!/usr/bin/env python3

# This script tests the rate of the isotropization of an electron distribution
# that initially has different temperatures along x and along y/z, in 3D or RZ.
# The decay of the temperature difference Tx - Ty, between the initial step and
# the step of the given plotfile, is compared with the analytical solution (see
# analysis_collision_3d_isotropization.py for references), integrated in time with
# the relaxation rate evaluated at the current temperatures.
#
# The simulation is stopped before the distribution becomes isotropic, so that
# the test is sensitive to the collision rate, and thus to the density computed
# within each collision bin (which depends on the volume of the bins).

import sys

import numpy as np
import scipy.constants as sc
import yt

sys.path.append("../../../Tools/Parser/")
from input_file_parser import parse_input_file

e = sc.e
pi = sc.pi
ep0 = sc.epsilon_0
m = sc.m_e

input_dict = parse_input_file("warpx_used_inputs")
dt = float(input_dict["warpx.const_dt"][0])
ne = float(input_dict["electron.density"][0])
log = float(input_dict["collision1.CoulombLog"][0])


def relaxation_rate(T_par, T_per):
    """Isotropization rate, for temperatures in Joules"""
    A = 1.0 - T_per / T_par
    return (
        e**4
        * ne
        * log
        / (8.0 * pi**1.5 * ep0**2 * m**0.5 * T_par**1.5)
        * A ** (-2)
        * (-3.0 + (3.0 - A) * np.arctanh(A**0.5) / A**0.5)
    )


def temperatures(fn):
    """Return the temperatures along x and y (in Joules), and the step number"""
    ds = yt.load(fn)
    ad = ds.all_data()
    ux = ad["electron", "particle_momentum_x"].to_ndarray() / m
    uy = ad["electron", "particle_momentum_y"].to_ndarray() / m
    # Use weighted averages, since the particle weights are not uniform in RZ geometry
    w = ad["electron", "particle_weight"].to_ndarray()
    Tx = np.average(ux**2, weights=w) * m
    Ty = np.average(uy**2, weights=w) * m
    return Tx, Ty, int(fn[-6:])


fn = sys.argv[1].rstrip("/")
Tx_sim, Ty_sim, nt = temperatures(fn)
Tx0, Ty0, _ = temperatures(fn[:-6] + "000000")

# Integrate the analytical solution, starting from the initial temperatures
# of the simulation (one collision per time step)
Tx, Ty = Tx0, Ty0
for _ in range(nt):
    mu = relaxation_rate(Tx, Ty)
    Tx, Ty = Tx + dt * mu * (Ty - Tx) * 2.0, Ty + dt * mu * (Tx - Ty)

decay_sim = (Tx_sim - Ty_sim) / (Tx0 - Ty0)
decay_theory = (Tx - Ty) / (Tx0 - Ty0)

tolerance = 0.05
error = abs(decay_sim - decay_theory) / decay_theory

print(
    f"decay of Tx - Ty after {nt} steps: simulation {decay_sim}, theory {decay_theory}"
)
print(f"error = {error}")
print(f"tolerance = {tolerance}")
assert error < tolerance
