#!/usr/bin/env python3
#
# --- Analysis script for the Darwin solver conducting (PEC) wall tests.
# --- The FieldProbe "Line" reduced diagnostic runs across the domain, from
# --- one wall to the other, and the resulting field profiles are checked for:
# ---   (a) the PEC condition - the tangential E-field (Ey) must vanish at the
# ---       walls;
# ---   (b) the response to the drive - the tangential B-field driven by the
# ---       Jy current sheet must follow the expected cos(pi*s/L) profile, with
# ---       the expected sign and with comparable amplitudes in the two halves
# ---       of the domain (the drive is symmetric about the domain midplane and
# ---       both walls get the same treatment, so a one-sided error in the wall
# ---       treatment shows up as a lopsided response);
# ---   (c) no runaway (unsaturated) growth of the field+particle energy by
# ---       the end of the run.

import dill
import numpy as np

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

probe_data = np.loadtxt("diags/line_probe.txt", skiprows=1)
step = probe_data[:, 0]
num_steps = len(np.unique(step))
resolution = len(np.where(step == step[0])[0])
probe_data = probe_data.reshape((num_steps, resolution, probe_data.shape[1]))

# sort probe points along the wall-normal coordinate
coord = probe_data[0, :, sim.coord_col]
order = np.argsort(coord)
coord = coord[order]
probe_data = probe_data[:, order, :]

assert np.all(np.isfinite(probe_data)), "Non-finite values in the field probe data"
assert coord[0] == 0.0 and np.isclose(coord[-1], sim.L_wall, rtol=1e-8), (
    "The field probe line does not span the domain from wall to wall"
)

# The scheme carries an undamped parasitic mode at the Nyquist frequency
# pi/dt (its field alternates sign every step); average the last two samples,
# one step apart, to remove it from the profiles checked below.
assert np.all(np.diff(np.unique(step))[-1:] == 1), "The probe must sample every step"
final = 0.5 * (probe_data[-1] + probe_data[-2])

# --- (a) PEC condition: the tangential E-field must vanish at the walls,
# --- relative to its interior scale. This is checked on the last sample
# --- (including the parasitic mode, which must satisfy it too) and on the
# --- two-sample average.
for label, profile in (("last sample", probe_data[-1]), ("average", final)):
    Ey = profile[:, sim.tangential_E_col]
    Ey_scale = np.max(np.abs(Ey[1:-1]))
    Ey_wall = max(abs(Ey[0]), abs(Ey[-1]))
    print(
        f"Tangential E at the walls ({label}): {Ey_wall:.3e} "
        f"(interior scale {Ey_scale:.3e})"
    )
    assert Ey_scale > 0.0, "The tangential E-field is identically zero"
    assert Ey_wall < 1e-8 * Ey_scale, (
        f"The tangential E-field does not vanish at the PEC walls ({label}): "
        f"wall value {Ey_wall:.3e} vs. interior scale {Ey_scale:.3e}"
    )

# --- (b) Driven B-field response: least-squares fit of the expected profile,
# --- over the full domain and over each half separately.
B = final[:, sim.driven_B_col]
shape = np.cos(np.pi * coord / sim.L_wall)
half = len(coord) // 2


def fit_amplitude(y, g):
    return np.dot(y, g) / np.dot(g, g)


amp = fit_amplitude(B, shape)
amp_left = fit_amplitude(B[:half], shape[:half])
amp_right = fit_amplitude(B[-half:], shape[-half:])
residual = np.sqrt(np.mean((B - amp * shape) ** 2)) / abs(amp)
asymmetry = abs(amp_left - amp_right) / abs(amp)
print(
    f"Driven B-field amplitude {amp:.3e} (left half {amp_left:.3e}, right half "
    f"{amp_right:.3e}), relative residual {residual:.2f}, asymmetry {asymmetry:.2f}"
)
assert np.sign(amp) == sim.driven_B_sign, "The driven B-field has the wrong sign"
assert residual < 1.0, "The driven B-field does not follow the expected profile"
assert asymmetry < 0.4, (
    "The driven B-field response is lopsided between the two halves of the domain"
)

# --- (c) No runaway (sustained, unsaturated) energy growth at the end of the
# --- run. The drift perturbation is an undamped free-energy source, but an
# --- error in the wall treatment produces sustained growth by large factors.
field_energy = np.loadtxt("diags/field_energy.txt", skiprows=1)
part_energy = np.loadtxt("diags/part_energy.txt", skiprows=1)
assert np.all(np.isfinite(field_energy)), "Non-finite field energy"
assert np.all(np.isfinite(part_energy)), "Non-finite particle energy"

total_energy = field_energy[:, 2] + part_energy[:, 2]
ratios = total_energy[1:] / total_energy[:-1]
tail = ratios[-max(2, len(ratios) // 3) :]
tail_growth = np.prod(tail) ** (1.0 / len(tail))  # geometric mean
print(f"Total energy growth per step at the end of the run: {tail_growth:.6f}")
assert tail_growth < 1.01, (
    f"The field+particle energy is still growing by a factor of {tail_growth:.4f} "
    "per step at the end of the run"
)

print("Darwin solver PEC boundary checks passed.")
