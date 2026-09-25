#!/usr/bin/env python3

# This script checks conservation of energy and charge for simulation
# of the free expansion of a plasma sphere in 1D RSPHERE geometry.
# Check energy before particle loss and final-step charge conservation away
# from radial boundaries.

import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0

tolerance_rel_energy = 2.0e-14
tolerance_rel_charge = 2.0e-14

pltdir = sys.argv[1]
rootdir = "./diags/"

field_energy_diag = np.loadtxt(rootdir + "/reduced_files/field_energy.txt", skiprows=1)
parts_energy_diag = np.loadtxt(
    rootdir + "/reduced_files/particle_energy.txt", skiprows=1
)
parts_number_diag = np.loadtxt(
    rootdir + "/reduced_files/particle_number.txt", skiprows=1
)

step = field_energy_diag[:, 0]
field_energy = field_energy_diag[:, 2]
parts_energy = parts_energy_diag[:, 2]
total_energy = field_energy + parts_energy

# Locate the last particle-number sample before particles reach the absorbing boundary.
diff_parts = parts_number_diag[:, 2] - parts_number_diag[0, 2]
idx = np.flatnonzero(~np.isclose(diff_parts, 0))[0] - 1

# Measure the relative change in total energy through that sample.
delta_E = (total_energy[idx] - total_energy[0]) / total_energy[0]
max_delta_E = np.abs(delta_E).max()

# Check energy conservation
print(f"max change in energy at idx = {idx}: {max_delta_E}")
print(f"energy tolerance: {tolerance_rel_energy:.6e}")
assert max_delta_E < tolerance_rel_energy

# Compare summed species charge density with epsilon0*divE on the final diagnostic step.
# Omit the first and last radial cells, which are adjacent to the boundaries.
ds = yt.load(pltdir)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

rho = (
    (data[("boxlib", "rho_electrons")] + data[("boxlib", "rho_ions")])
    .to_ndarray()
    .squeeze()
)
divE = data[("boxlib", "divE")].to_ndarray().squeeze()

assert rho.ndim == 1 and divE.ndim == 1, "Expected 1D RSPHERE diagnostic fields"
assert rho.size > 2, "Need interior cells after excluding boundary-adjacent cells"

charge_error = rho[1:-1] - epsilon_0 * divE[1:-1]
# Normalize to the characteristic charge density e*n0 from the input deck.
n0 = 1.0e22
error_rms = np.sqrt(np.mean((charge_error / (e * n0)) ** 2))

print(f"final-step interior RMS charge-conservation error: {error_rms:.6e}")
print(f"charge tolerance: {tolerance_rel_charge:.6e}")
assert error_rms < tolerance_rel_charge
