#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Full radiation/PIC ownership accounting with prescribed lab opacity.

This is a numerical carry integration gate, not a Doppler or moving-LTE test.
"""

import argparse
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(40)
C = np.longdouble("299792458")
QE = np.longdouble("1.602176634e-19")
KB = np.longdouble("1.380649e-23")


def table(path):
    with path.open() as stream:
        labels = [
            token.split("]", 1)[1] for token in stream.readline().lstrip("#").split()
        ]
    rows = np.atleast_2d(np.loadtxt(path))
    assert rows.shape[1] == len(labels)
    assert np.all(np.isfinite(rows))
    return dict(zip(labels, rows.T, strict=True))


def state(path):
    ds = yt.load(str(path))
    data = ds.all_data()
    ids = np.asarray(data["ions", "particle_id"].v, dtype=np.int64)
    cpu = np.asarray(data["ions", "particle_cpu"].v, dtype=np.int64)
    order = np.lexsort((cpu, ids))
    assert len(order) == int(np.prod(ds.domain_dimensions))

    def particle(name):
        return np.asarray(data["ions", f"particle_{name}"].v, dtype=np.longdouble)[
            order
        ]

    velocity = np.stack(
        [
            np.asarray(
                data["ions", f"particle_momentum_{a}"].to_value("kg*m/s"),
                dtype=np.longdouble,
            )[order]
            for a in "xyz"
        ],
        axis=1,
    )  # The declared ion rest mass is exactly 1 kg.
    carry = np.stack(
        [particle(f"radiation_impulse_streaming_u{a}") for a in "xyz"], axis=1
    )
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    rho = np.asarray(grid["boxlib", "rho_ions"].v, dtype=np.longdouble)
    temperature = np.asarray(grid["boxlib", "Te"].v, dtype=np.longdouble)
    volume = np.prod(np.asarray(ds.domain_width.v, dtype=np.longdouble)) / np.prod(
        ds.domain_dimensions
    )
    internal = np.sum(rho / QE * KB * temperature * np.longdouble("1.5")) * volume
    return {
        "id": np.stack((ids[order], cpu[order]), axis=1),
        "u": velocity,
        "w": particle("weight"),
        "x": particle("position_x" if ds.dimensionality == 1 else "position_y"),
        "carry": carry,
        "work": particle("radiation_impulse_streaming_work"),
        "internal": internal,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
energy = table(Path("diags/radiation_energy.txt"))
momentum = table(Path("diags/radiation_momentum.txt"))
steps = energy["step()"].astype(int)
np.testing.assert_array_equal(steps, momentum["step()"].astype(int))
assert steps[-1] == 32
assert steps[0] == (16 if args.reference is not None else 0)
np.testing.assert_array_equal(steps, np.arange(steps[0], 33))
root = args.reference if args.reference is not None else Path(".")
initial = state(root / "diags/diag000000")
initial_energy = np.longdouble(1000)
scale_p = initial_energy / C
alpha = np.longdouble("9.314526130745389e-1")
np.testing.assert_allclose(
    energy["total_radiation(J)"],
    initial_energy * np.exp(-alpha * C * energy["time(s)"]),
    rtol=1.0e-11,
    atol=1.0e-11,
)
pending = energy["pending_material_carry_energy(J)"]
assert np.max(np.abs(pending)) > 1.0e-3
assert np.max(np.abs(momentum["pending_streaming_material_z(kg*m/s)"])) > 1.0e-10
assert np.max(energy["material_kinetic_exchange(J)"]) > 1
np.testing.assert_allclose(
    energy["material_exchange(J)"],
    (
        energy["material_internal_exchange(J)"]
        + energy["material_kinetic_exchange(J)"]
        + energy["material_carry_energy_change(J)"]
    ),
    rtol=1.0e-12,
    atol=1.0e-11,
)
old_pending = 0.0
if args.reference is not None:
    baseline = table(args.reference / "diags/radiation_energy.txt")
    old_pending = baseline["pending_material_carry_energy(J)"][15]
np.testing.assert_allclose(
    pending - np.r_[old_pending, pending[:-1]],
    energy["material_carry_energy_change(J)"],
    rtol=1.0e-11,
    atol=1.0e-11,
)
np.testing.assert_allclose(
    energy["total_radiation(J)"]
    + energy["cumulative_material_exchange(J)"]
    + energy["cumulative_numerical_energy_residual(J)"],
    initial_energy,
    rtol=0,
    atol=float(1.0e-10 * initial_energy),
)
worst = 0.0
previous = None
for row, step in enumerate(steps):
    current = state(Path(f"diags/diag{step:06d}"))
    np.testing.assert_array_equal(current["id"], initial["id"])
    np.testing.assert_array_equal(current["w"], initial["w"])
    du = current["u"] - initial["u"]
    old_gamma = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
    new_gamma = np.sqrt(1 + np.sum(current["u"] ** 2, axis=1) / C**2)
    kinetic = np.sum(
        current["w"]
        * np.sum((current["u"] + initial["u"]) * du, axis=1)
        / (old_gamma + new_gamma)
    )
    carry_work = np.sum(current["w"] * current["work"])
    np.testing.assert_allclose(carry_work, pending[row], rtol=1.0e-11, atol=1.0e-11)
    impulse = np.sum(current["w"][:, None] * (du + current["carry"]), axis=0)
    expected = (initial_energy - energy["total_radiation(J)"][row]) / C
    np.testing.assert_allclose(
        impulse, [0, 0, expected], rtol=0, atol=float(1.0e-10 * scale_p)
    )
    balance = (
        energy["total_radiation(J)"][row]
        + kinetic
        + carry_work
        + current["internal"]
        - initial["internal"]
        - initial_energy
    )
    error = float(abs(balance) / initial_energy)
    worst = max(worst, error)
    assert error < 1.0e-10, (step, "actual radiation/ion/electron/carry balance", error)
    if previous is not None:
        moved = current["x"] - previous["x"]
        expected_move = current["u"][:, 2] / new_gamma * np.longdouble("1.e-9")
        periodic_error = (
            moved - expected_move + np.longdouble("0.5")
        ) % 1 - np.longdouble("0.5")
        assert np.max(np.abs(periodic_error)) < 1.0e-11
    previous = current

if args.reference is not None:
    baseline = table(args.reference / "diags/radiation_energy.txt")
    for name in [
        "total_radiation(J)",
        "cumulative_material_exchange(J)",
        "pending_material_carry_energy(J)",
    ]:
        np.testing.assert_allclose(
            energy[name], baseline[name][steps], rtol=1.0e-11, atol=1.0e-10
        )
    baseline_state = state(args.reference / "diags/diag000032")
    np.testing.assert_array_equal(previous["u"], baseline_state["u"])
    np.testing.assert_allclose(
        previous["carry"], baseline_state["carry"], rtol=1.0e-10, atol=1.0e-18
    )
    np.testing.assert_allclose(
        previous["work"], baseline_state["work"], rtol=1.0e-10, atol=1.0e-10
    )

print(
    f"Moving particle carry with full radiation/PIC exchange: worst actual balance={worst:.3e}"
)
