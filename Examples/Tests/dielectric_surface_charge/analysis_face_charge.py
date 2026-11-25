#!/usr/bin/env python3

# Electrons, uniform along one axis, move along the other axis and are absorbed
# by a dielectric, after crossing a domain boundary (reflecting or periodic)
# in the same time step as the dielectric surface. This checks that:
# - all electrons are absorbed, and their charge is conserved in the surface charge
# - the surface charge is deposited on the surface where the electrons entered the dielectric
#
# Usage: analysis_face_charge.py <plotfile> <axis (x or z)> <position of the surface>

import sys

import numpy as np
import yt
from scipy.constants import e

yt.funcs.mylog.setLevel(50)

filename = sys.argv[1]
axis = {"x": 0, "z": 1}[sys.argv[2]]
face = float(sys.argv[3])

# Particle weights, as a function of time
# columns: step, time, total_macroparticles, electrons_macroparticles,
#          total_weight, electrons_weight
particle_number = np.loadtxt("diags/reducedfiles/particle_number.txt")
electrons_weight = particle_number[:, 5]

# All electrons have been absorbed by the end of the simulation
assert electrons_weight[0] > 0.0
assert electrons_weight[-1] == 0.0

ds = yt.load(filename)
grid = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
surface_charge = grid["boxlib", "dielectric_surface_charge"].v.squeeze()
dx, dz = (ds.domain_width / ds.domain_dimensions).v[:2]
lo = ds.domain_left_edge.v[axis]
d = (dx, dz)[axis]
n = ds.domain_dimensions[axis]
periodic = ds.periodicity[axis]

# Charge conservation: the charge of the absorbed electrons is in the surface charge
# (in 2D, both are per unit length along y)
absorbed_charge = -e * electrons_weight[0]
total_surface_charge = np.sum(surface_charge) * dx * dz
print(f"absorbed charge:      {absorbed_charge:.10e} C/m")
print(f"total surface charge: {total_surface_charge:.10e} C/m")
assert np.isclose(total_surface_charge, absorbed_charge, rtol=1.0e-10, atol=0.0)

# Expected profile along the axis, if all the charge is on the surface:
# linear deposition on the nodes, averaged to the cell centers in the plotfile
s = (face - lo) / d
i = int(np.floor(s))
w = s - i
nodal = np.zeros(n + 1)
nodal[i] += 1.0 - w
nodal[i + 1] += w
if periodic:
    nodal[0] += nodal[n]
    nodal[n] = nodal[0]
expected = 0.5 * (nodal[:-1] + nodal[1:])
expected /= expected.sum()

other_axis = 1 - axis
profile = surface_charge.sum(axis=other_axis) / surface_charge.sum()
cells = np.nonzero((np.abs(profile) > 1.0e-12) | (expected > 0.0))[0]
print("cells:                 ", cells)
print("surface charge profile:", np.round(profile[cells], 6))
print("expected profile:      ", np.round(expected[cells], 6))
assert np.allclose(profile, expected, rtol=0.0, atol=1.0e-6)
