#!/usr/bin/env python3
"""Analysis for test_2d_particle_boundary_interaction_thermal.

A fully accommodating (diffuse) embedded-boundary wall at 600 K re-emits each impacting particle
from a half-Maxwellian at the wall temperature. Starting from a 300 K gas, repeated wall strikes
heat the gas toward the wall temperature. The wall still re-emits (does not absorb), so particle
number is conserved while the mean kinetic temperature rises from 300 K toward 600 K.

Reads the ParticleNumber and ParticleEnergy reduced diagnostics; asserts number conservation and
monotone-ish heating toward, but not exceeding, the wall temperature.
"""

import numpy as np

kB = 1.380649e-23

npart = np.loadtxt("diags/reducedfiles/npart.txt")
pen = np.loadtxt("diags/reducedfiles/pen.txt")

N = npart[:, 2]  # macroparticle count (constant unless the wall absorbs)
mean_ke = pen[:, 4]  # mean kinetic energy per real particle [J] (total_mean column)

# mean kinetic energy per real particle = (3/2) kB T  (particles carry 3 velocity components)
T = (2.0 / 3.0) * mean_ke / kB
T0, T1 = T[0], T[-1]

N_spread = (N.max() - N.min()) / N[0]
print(f"particle number: {N[0]:.6e} -> {N[-1]:.6e}   (max spread {N_spread:.3e})")
print(f"kinetic temperature: {T0:.1f} K -> {T1:.1f} K   (gas 300 K, wall 600 K)")

# re-emission, not absorption
assert N_spread < 5e-3, (
    "particle number not conserved -> EB is absorbing/leaking, not re-emitting"
)
# gas heats toward the wall ...
assert T1 > T0 + 50.0, "gas did not heat -> thermal accommodation not applied"
assert T1 > 380.0, "insufficient heating toward the 600 K wall"
# ... but never exceeds the wall temperature
assert T1 < 610.0, (
    "gas overshot the wall temperature -> thermal accommodation is unphysical"
)

print(
    "PASS: thermal EB re-emission conserves particle number and thermalizes the gas toward the wall."
)
