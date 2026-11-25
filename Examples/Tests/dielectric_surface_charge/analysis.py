#!/usr/bin/env python3

# A sheet of electrons is absorbed on the surface of a dielectric slab.
# This checks that:
# - the protons initialized inside the slab are removed at injection
# - all electrons are absorbed, and their charge is conserved in the surface charge
# - the surface charge is localized at the surface of the slab
# - the potential matches the analytical solution for a charged sheet on the
#   surface of a dielectric slab, between two grounded plates

import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0

yt.funcs.mylog.setLevel(50)

# Parameters (from the inputs file)
z_lo, z_hi = 0.0, 1.0e-2
z_slab_lo, z_slab_hi = 5.0e-3, 7.0e-3
epsilon_r = 4.0

filename = sys.argv[1]
# Optional: position of the lower surface of the slab, if it differs from the base inputs file
if len(sys.argv) > 2:
    z_slab_lo = float(sys.argv[2])

# Particle weights, as a function of time
# columns: step, time, total_macroparticles, electrons_macroparticles,
#          protons_macroparticles, total_weight, electrons_weight, protons_weight
particle_number = np.loadtxt("diags/reducedfiles/particle_number.txt")
electrons_weight = particle_number[:, 6]
protons_weight = particle_number[:, 7]

# No protons are ever present: they are all initialized inside the slab
assert np.all(protons_weight == 0.0)
# All electrons have been absorbed by the end of the simulation
assert electrons_weight[0] > 0.0
assert electrons_weight[-1] == 0.0

# Fields at the end of the simulation
ds = yt.load(filename)
grid = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
surface_charge = grid["boxlib", "dielectric_surface_charge"].v.squeeze()
phi = grid["boxlib", "phi"].v.squeeze()
dx, dz = (ds.domain_width / ds.domain_dimensions).v[:2]
lx = ds.domain_width.v[0]
nz = ds.domain_dimensions[1]
z = z_lo + (np.arange(nz) + 0.5) * dz

# Charge conservation: the charge of the absorbed electrons is in the surface charge
# (in 2D, both are per unit length along y)
absorbed_charge = -e * electrons_weight[0]
total_surface_charge = np.sum(surface_charge) * dx * dz
print(f"absorbed charge:      {absorbed_charge:.10e} C/m")
print(f"total surface charge: {total_surface_charge:.10e} C/m")
assert np.isclose(total_surface_charge, absorbed_charge, rtol=1.0e-10, atol=0.0)

# The surface charge is localized at the surface of the slab
# (within the particle shape, plus the cell-centering of the nodal field)
far_from_surface = np.abs(z - z_slab_lo) > 2.0 * dz
assert np.all(surface_charge[:, far_from_surface] == 0.0)

# Analytical potential for a charge sheet of areal density sigma at z = z_slab_lo:
# the electric displacement D is uniform in each region, jumps by sigma across the sheet,
# and the potential vanishes on both plates.
sigma = total_surface_charge / lx
l1 = z_slab_lo - z_lo
l2 = z_slab_hi - z_slab_lo
l3 = z_hi - z_slab_hi
a = l2 / epsilon_r + l3
D1 = -sigma * a / (l1 + a)
E1 = D1 / epsilon_0
E2 = (D1 + sigma) / (epsilon_r * epsilon_0)
E3 = (D1 + sigma) / epsilon_0
phi1 = -E1 * l1
phi2 = phi1 - E2 * l2
phi_exact = np.where(
    z < z_slab_lo,
    -E1 * (z - z_lo),
    np.where(z < z_slab_hi, phi1 - E2 * (z - z_slab_lo), phi2 - E3 * (z - z_slab_hi)),
)

# The problem is uniform in x
phi_z = phi.mean(axis=0)
assert np.allclose(phi, phi_z[np.newaxis, :], rtol=0.0, atol=1.0e-6 * np.abs(phi1))

error = np.max(np.abs(phi_z - phi_exact)) / np.max(np.abs(phi_exact))
print(f"potential at the surface: {phi1:.6e} V")
print(f"max relative error of the potential: {error:.6e}")
assert error < 0.02
