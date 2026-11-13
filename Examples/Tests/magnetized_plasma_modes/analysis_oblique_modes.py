#!/usr/bin/env python3
"""Analysis for inputs_test_3d_darwin_solver_oblique_modes_picmi.py.

Reduced (16^3, 130 steps) version of a longer verification run (32^3, ~4700
steps) in which every low-k (k l_i < 0.9) transverse branch peaked within 1-2
frequency bins of cold two-fluid theory. The record here (5.2 W_ci^-1, about
half a wave period) is too short to measure frequencies, so this script checks:

1. Permutation symmetry: the permutation-symmetric particle loading makes the
   members of each wave-vector group equivalent, so their mean |dB(k)|^2
   should agree to roundoff (a mixed-up mass-matrix index would break this).
2. Energy conservation: max |dE/E0| over the run.

Measured on 16 noise seeds (CI uses seed 1): power ratio - 1 < 2e-14 for both
groups, max |dE/E0| = 1.3e-4 to 1.6e-4.
"""

import numpy as np

# max allowed (max/min - 1) of the mean |dB(k)|^2 within a permutation group
SYMMETRY_TOL = 1e-6
ENERGY_TOL = 5e-4  # max allowed |dE/E0| over the run

d = np.load("diags/k_modes.npz")
amp = d["amp"]  # (nt, nk, 3) complex
k_list = [tuple(int(x) for x in m) for m in d["k_list"]]

# --------------------------------------------------------------- symmetry
groups = {}
for j, m in enumerate(k_list):
    key = tuple(sorted(abs(x) for x in m))
    groups.setdefault(key, []).append(j)

for key, members in sorted(groups.items()):
    pows = [np.mean(np.sum(np.abs(amp[:, j, :]) ** 2, axis=1)) for j in members]
    pow_ratio_m1 = max(pows) / min(pows) - 1.0
    print(
        f"group {[k_list[j] for j in members]}: power ratio - 1 (max/min) = "
        f"{pow_ratio_m1:.2e} (tol {SYMMETRY_TOL:g})"
    )
    assert pow_ratio_m1 <= SYMMETRY_TOL


# --------------------------------------------------------------- energy
def tot(path):
    with open(path) as f:
        header = f.readline().split()
    data = np.loadtxt(path)
    # FieldEnergy totals are "total_lev0(J)"; ParticleEnergy has "total(J)" (and a
    # per-particle "total_mean(J)" that must not be added)
    labels = [x.split("]")[-1] for x in header]
    idx = [i for i, x in enumerate(labels) if x in ("total_lev0(J)", "total(J)")]
    return data[:, idx].sum(axis=1)


fE = tot("diags/field_energy.txt")
pE = tot("diags/part_energy.txt")
E = fE + pE
max_dE = np.max(np.abs(E - E[0]) / E[0])
print(
    f"energy: max|dE/E0|={max_dE:.3e} (tol {ENERGY_TOL:g}), "
    f"final={(E[-1] - E[0]) / E[0]:+.3e}"
)
assert max_dE <= ENERGY_TOL
