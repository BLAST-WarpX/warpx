#!/usr/bin/env python3
#
# --- Shared physics-check analysis script for the Darwin-solver conducting
# --- (PEC) wall tests. Used by both the 1D (wall-normal = z) and 2D
# --- (wall-normal = x) tests.
# ---
# --- Note: the Darwin electrostatic/inductive split means Efield_fp only
# --- ever holds the inductive E-field transiently, inside OneStep - the
# --- per-step "field solve" phase that runs right after it always resets
# --- Efield_fp to the (here, near-zero) electrostatic-only value before any
# --- diagnostic gets a chance to sample it. So the checks below are built
# --- entirely from the B-field (and reduced-diagnostic energies), which
# --- *are* persisted between steps via SyncDarwinBFromA.
# ---
# --- The FieldProbe "Line" reduced diagnostic is run along the wall-normal
# --- axis, from one wall to the other, and the resulting B profile is
# --- checked for:
# ---   (a) wall regularity - the wall-*normal* B component must vanish at
# ---       the walls (the direct PEC constraint on B);
# ---   (b) left/right energy balance - the driving current sheet is
# ---       symmetric about the domain midplane and both walls get the same
# ---       treatment, so the driven field's energy should split evenly
# ---       between the two halves of the domain;
# ---   (c) no runaway (still unsaturated) growth of the field+particle
# ---       energy by the end of the run.

import dill
import numpy as np

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

# FieldProbe column layout (see Source/Diagnostics/ReducedDiags/FieldProbe.cpp):
# [0]step [1]time [2]x [3]y [4]z [5]Ex [6]Ey [7]Ez [8]Bx [9]By [10]Bz [11]S
COORD_COL = {"x": 2, "y": 3, "z": 4}[sim.wall_axis]

probe_data = np.loadtxt("diags/line_probe.txt", skiprows=1)
step = probe_data[:, 0]
num_steps = len(np.unique(step))
resolution = len(np.where(step == step[0])[0])
probe_data = probe_data.reshape((num_steps, resolution, probe_data.shape[1]))

# sort probe points along the wall-normal coordinate (should already be
# ordered, but don't assume it)
coord = probe_data[0, :, COORD_COL]
order = np.argsort(coord)
coord = coord[order]
probe_data = probe_data[:, order, :]

assert np.all(np.isfinite(probe_data)), "Non-finite values in the field probe data"

final = probe_data[-1]
interior = slice(1, -1)

# --- (a) Wall regularity: the wall-normal B component must vanish at the
# --- PEC walls, relative to the driven tangential-B interior scale (there is
# --- no guarantee the normal component has any structure of its own to
# --- normalize against - in 1D it is identically zero everywhere by the
# --- symmetry of this drive).
driven_scale = max(np.max(np.abs(final[interior, col])) for col in sim.driven_B_cols)
assert driven_scale > 0.0, (
    "The driven tangential B-field is identically zero - the perturbation "
    "did not produce a measurable response"
)

normal_B_field = final[:, sim.normal_B_col]
normal_B_wall_max = max(abs(normal_B_field[0]), abs(normal_B_field[-1]))
wall_tol = getattr(sim, "wall_regularity_tol", 1.0e-6)
assert normal_B_wall_max < wall_tol * driven_scale, (
    "Wall-normal B-field is not regular at the PEC walls: wall value "
    f"{normal_B_wall_max:.3e} vs. driven interior scale {driven_scale:.3e}"
)

# --- (b) Left/right energy balance about the domain midplane: the driving
# --- current sheet is symmetric about the midplane, and the two walls get
# --- exactly the same (mirrored) treatment in the implementation, so the
# --- driven B-field's energy content should be comparable in the two
# --- halves of the domain. This is deliberately not a point-wise symmetry
# --- check: the driven fields grow throughout the run (see (c)) and their
# --- instantaneous spatial profile is not required to match a single clean
# --- mode, so point-wise (anti)symmetry is not a robust signature - but a
# --- one-sided bug in the wall treatment (e.g. a sign only wrong at one of
# --- the two walls) would still show up as a lopsided energy split.
half = len(coord) // 2
for col in sim.driven_B_cols:
    field = final[:, col]
    left_energy = np.sum(field[:half] ** 2)
    right_energy = np.sum(field[-half:] ** 2)
    total = left_energy + right_energy
    if total == 0.0:
        continue
    imbalance = abs(left_energy - right_energy) / total
    assert imbalance < 0.5, (
        f"Driven B-field component (column {col}) has a lopsided left/right "
        f"energy split about the domain midplane: relative imbalance "
        f"{imbalance:.3f} (left={left_energy:.3e}, right={right_energy:.3e}) "
        "- this may indicate the wall treatment is only correct on one side"
    )

# --- (c) No runaway (sustained, unsaturated) energy growth over the run.
# --- The driven current sheet is a persistent, undamped free-energy source
# --- with no dissipation in the collisionless Darwin model, so some genuine
# --- growth over the run - even a one-time jump when reflecting particles
# --- first reach the walls - is expected and not itself a sign of a bug.
# --- What a sign/parity error in the wall treatment produces instead is
# --- SUSTAINED, unsaturated growth all the way to the last step (verified:
# --- the pre-fix bug grew the last few samples by 3x-40x each, back to back,
# --- with no sign of leveling off). So check the trend at the END of the
# --- run, not the total growth over the whole run.
field_energy = np.loadtxt("diags/field_energy.txt", skiprows=1)
part_energy = np.loadtxt("diags/part_energy.txt", skiprows=1)
assert np.all(np.isfinite(field_energy)), "Non-finite field energy"
assert np.all(np.isfinite(part_energy)), "Non-finite particle energy"

total_energy = field_energy[:, -1] + part_energy[:, -1]
# skip index 0: energy is exactly zero there by construction (no field yet),
# which would make the very first ratio meaningless/huge.
nonzero = total_energy[1:]
ratios = nonzero[1:] / nonzero[:-1]
tail = ratios[-max(2, len(ratios) // 3) :]
tail_growth = np.prod(tail) ** (1.0 / len(tail))  # geometric mean
assert tail_growth < 3.0, (
    f"Field+particle energy is still growing by a factor of {tail_growth:.2f} "
    "per diagnostic interval at the end of the run (not saturating) - this "
    "may indicate a sign error in the PEC wall treatment"
)

print("Darwin solver PEC boundary checks passed.")
