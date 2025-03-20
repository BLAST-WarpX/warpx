#!/usr/bin/env python3

# Run the default regression test for the PICMI version of the EB test
# using the same reference file as for the non-PICMI test since the two
# tests are otherwise the same.

# Check reduced diagnostics for charge on EB
import numpy as np
from scipy.constants import epsilon_0

# Theoretical charge on the embedded boundary, for sphere at potential phi_0
phi_0 = 1.0  # V
R = 0.1  # m
q_th = 4 * np.pi * epsilon_0 * phi_0 * R
print("Theoretical charge: ", q_th)

data = np.loadtxt("diags/reducedfiles/eb_charge.txt")
q_sim = data[1, 2]
print("Simulation charge: ", q_sim)
assert abs((q_sim - q_th) / q_th) < 0.06

data_eighth = np.loadtxt("diags/reducedfiles/eb_charge_one_eighth.txt")
q_sim_eighth = data_eighth[1, 2]
assert abs((q_sim_eighth - q_th / 8) / (q_th / 8)) < 0.06
