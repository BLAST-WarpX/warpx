#!/usr/bin/env python3

#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the scripts `inputs_test_2d_pec_field_insulator_implicit` and
# `inputs_test_2d_pec_field_insulator_implicit_restart`.
# The scripts model an insulator boundary condition on part of the
# upper x boundary that pushes B field into the domain. The implicit
# solver is used, converging to machine tolerance. The energy accounting
# should be exact to machine precision, so that the energy is the system
# should be the same as the amount of energy pushed in from the boundary.
# This is checked using the FieldEnergy and FieldPoyntingFlux reduced
# diagnostics.
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# this will be the name of the plot file
fn = sys.argv[1]

EE = np.loadtxt(f"{fn}/../reducedfiles/fieldenergy.txt", skiprows=1)
SS = np.loadtxt(f"{fn}/../reducedfiles/poyntingflux.txt", skiprows=1)
SSsum = SS[:, 2:6].sum(1)
EEloss = SS[:, 7:].sum(1)

dt = EE[1, 1]

fig, ax = plt.subplots()
ax.plot(EE[:, 0], EE[:, 2], label="field energy")
ax.plot(SS[:, 0], -EEloss, label="-flux*dt")
ax.legend()
ax.set_xlabel("time (s)")
ax.set_ylabel("energy (J)")
fig.savefig("energy_history.png")

fig, ax = plt.subplots()
ax.plot(EE[:, 0], (EE[:, 2] + EEloss) / EE[:, 2].max())
ax.set_xlabel("time (s)")
ax.set_ylabel("energy difference/max energy (1)")
fig.savefig("energy_difference.png")

tolerance_rel = 1.0e-13

energy_difference_fraction = np.abs((EE[:, 2] + EEloss) / EE[:, 2].max()).max()
print(f"energy accounting error = {energy_difference_fraction}")
print(f"tolerance_rel = {tolerance_rel}")

assert energy_difference_fraction < tolerance_rel
