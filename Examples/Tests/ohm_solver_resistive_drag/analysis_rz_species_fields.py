#!/usr/bin/env python3

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

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

print("PASS")
