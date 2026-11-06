#!/usr/bin/env python3

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c

# Must match inputs_test_rz_ohm_solver_resistive_drag_species_fields_picmi.py.
# PICMI's directed_velocity is a proper velocity, so the lab-frame drift the
# deposits sample is u/gamma.
u_drift = 1.0e5
v_drift = u_drift / np.sqrt(1.0 + (u_drift / c) ** 2)

ts = OpenPMDTimeSeries("diags/field_diags")
iteration = ts.iterations[-1]

rho_species, _ = ts.get_field(field="rho_ions", iteration=iteration)
rho_species_sum, _ = ts.get_field(
    field="hybrid_rho_species_sum_fp", iteration=iteration
)

# openPMD arrays are (z, r). The outer radial boundary is excluded because the
# internal coupling field and a diagnostic-time particle deposit apply their
# boundary fills at different points. Everywhere else they represent the same
# physical species charge density.
rho_species = rho_species[:, :-1]
rho_species_sum = rho_species_sum[:, :-1]

scale = np.maximum(np.abs(rho_species), np.finfo(float).tiny)
relative_error = np.abs(rho_species_sum - rho_species) / scale

median_error = np.median(relative_error)
max_error = np.max(relative_error)
axis_error = np.max(relative_error[:, 0])

print("RZ per-species physical-density analysis")
print(f"  median relative error = {median_error:.4e}")
print(f"  maximum relative error = {max_error:.4e}")
print(f"  axis relative error = {axis_error:.4e}")

assert median_error < 1.0e-10
assert max_error < 1.0e-8
assert axis_error < 1.0e-8

# The ions carry a uniform axial drift, so the deposited per-species current
# is finite and, with a single charged species, must equal the total current
# density deposited by the plasma. This checks the RZ inverse-volume scaling
# of J_s: a missing or doubled scaling would be an O(1) error at every
# radius. Off the axis the two fields must agree to round-off. The on-axis
# cells are excluded here because they deliberately differ: the total j uses
# the standard (Verboncoeur) on-axis volume factor of the field solve, while
# J_s shares the rho_s deposit convention so that the reconstructed bulk
# velocity V_s = J_s / rho_s (consumed by the resistive drag and the
# electron-energy sources) is exact on the axis -- which is asserted next.
jz, _ = ts.get_field(field="j", coord="z", iteration=iteration)
jz_species, _ = ts.get_field(field="current_fp_ions[dir=z]", iteration=iteration)

jz_off = jz[:, 1:-1]
jz_species_off = jz_species[:, 1:-1]

scale = np.maximum(np.abs(jz_off), np.finfo(float).tiny)
relative_error_j = np.abs(jz_species_off - jz_off) / scale

median_error_j = np.median(relative_error_j)
max_error_j = np.max(relative_error_j)

print("RZ per-species current-density analysis")
print(f"  median relative error = {median_error_j:.4e}")
print(f"  maximum relative error = {max_error_j:.4e}")

assert median_error_j < 1.0e-10
assert max_error_j < 1.0e-8

# On-axis (and everywhere else): J_s and rho_s must share the same volume
# scaling, i.e. the bulk velocity J_s / rho_s of the uniformly drifting ions
# must equal the drift everywhere, including the axis cells. An inconsistent
# axis factor in either deposit would break this by O(1).
rho_species_full, _ = ts.get_field(
    field="hybrid_rho_species_sum_fp", iteration=iteration
)
vz = jz_species[:, :-1] / rho_species_full[:, :-1]
relative_error_v = np.abs(vz - v_drift) / v_drift

median_error_v = np.median(relative_error_v)
max_error_v = np.max(relative_error_v)
axis_error_v = np.max(relative_error_v[:, 0])

print("RZ per-species bulk-velocity (J_s/rho_s) analysis")
print(f"  median relative error = {median_error_v:.4e}")
print(f"  maximum relative error = {max_error_v:.4e}")
print(f"  axis relative error = {axis_error_v:.4e}")

assert median_error_v < 1.0e-8
assert max_error_v < 1.0e-8
assert axis_error_v < 1.0e-8

print("PASS")
