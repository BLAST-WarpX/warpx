#!/usr/bin/env python3

# Run the default regression test for the PICMI version of the EB test
# using the same reference file as for the non-PICMI test since the two
# tests are otherwise the same.

# Check reduced diagnostics for charge on EB, and check phi against the
# analytical solution for two concentric spherical EBs.

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import epsilon_0

# Geometry: inner sphere at phi=phi_0 and outer sphere at phi=0
phi_0 = 1.0  # V
R_inner = 0.1  # m
R_outer = 0.5  # m

# Theoretical charge on the inner sphere for two concentric spheres
# (inner at phi_0, outer grounded):
#   q = 4*pi*eps0 * phi_0 * R_inner*R_outer / (R_outer - R_inner)
q_th = 4 * np.pi * epsilon_0 * phi_0 * R_inner * R_outer / (R_outer - R_inner)
print("Theoretical charge: ", q_th)

# The ChargeOnEB reduced diagnostic evaluates E at the grid node just outside
# the EB (at r ~ R_inner + dx), not on the EB surface, which produces a
# systematic underestimate of the integrated charge on a fine grid. We use a
# loose tolerance here; the phi check below is the primary accuracy test.
data = np.loadtxt("diags/reducedfiles/eb_charge.txt")
q_sim = data[1, 2]
print("Simulation charge: ", q_sim)
assert abs((q_sim - q_th) / q_th) < 0.25

data_eighth = np.loadtxt("diags/reducedfiles/eb_charge_one_eighth.txt")
q_sim_eighth = data_eighth[1, 2]
assert abs((q_sim_eighth - q_th / 8) / (q_th / 8)) < 0.25

ts = OpenPMDTimeSeries("diags/diag1")

# Check that the eb_covered field is correct.
# With two concentric spherical EBs, eb_covered is 1 inside the inner sphere
# AND outside the outer sphere, and 0 in the spherical shell between them.
eb_covered, info = ts.get_field("eb_covered", iteration=0)
r = np.sqrt(
    info.x[:, np.newaxis, np.newaxis] ** 2
    + info.y[np.newaxis, :, np.newaxis] ** 2
    + info.z[np.newaxis, np.newaxis, :] ** 2
)
assert np.all(eb_covered >= 0)
assert np.all(eb_covered <= 1)
assert np.all(eb_covered[r < R_inner - info.dx] == 1)
assert np.all(eb_covered[r > R_outer + info.dx] == 1)
shell_mask = (r > R_inner + info.dx) & (r < R_outer - info.dx)
assert np.all(eb_covered[shell_mask] == 0)

# Check phi against the analytical solution phi(r) = A + B/r between the
# two concentric spheres. This is the exact solution to Laplace's equation
# with spherical symmetry.
B = 1.0 / (1.0 / R_inner - 1.0 / R_outer)
A = -B / R_outer
phi, info = ts.get_field("phi", iteration=0)
phi_theory = A + B / r
# Only compare cells well away from both EB surfaces (avoid cut cells)
mask = (r > R_inner + 2 * info.dx) & (r < R_outer - 2 * info.dx)
phi_error = np.max(np.abs(phi[mask] - phi_theory[mask]) / np.abs(phi_theory[mask]))
print(f"max relative error of phi: {phi_error}")
assert phi_error < 0.03
