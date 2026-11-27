#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent geometric-optics oracle for Cartesian mixed photon faces."""

import argparse
from pathlib import Path

import numpy as np
import yt
from scipy.constants import c, electron_mass

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--particle-precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--dims", choices=["1", "2", "3"], required=True)
parser.add_argument("--initial", default="diags/plt000000")
parser.add_argument("--absorption-coefficient", type=float, default=0.0)
args = parser.parse_args()
assert np.isfinite(args.absorption_coefficient) and args.absorption_coefficient >= 0
eps = max(
    np.finfo(np.float32 if precision == "SINGLE" else np.float64).eps
    for precision in (args.precision, args.particle_precision)
)
yt.set_log_level(50)


def yt_photon_energy(data):
    return np.sum(
        data["photons", "particle_weight"].v
        * c
        * np.sqrt(
            sum(data["photons", "particle_momentum_" + axis].v ** 2 for axis in "xyz")
        )
    )


def input_photon_energy(weight, u_dimless):
    return weight * electron_mass * c**2 * np.linalg.norm(u_dimless)


def trace(pos, direction, t_end, lo, hi, absorb_lo, absorb_hi):
    """Vacuum specular-or-escape trajectory. Does not call WarpX."""
    pos = np.array(pos, dtype=float)
    n = np.array(direction, dtype=float)
    n = n / np.linalg.norm(n)
    t = 0.0
    escaped = False
    ndim = lo.size
    while t < t_end and not escaped:
        hits = []
        for d in range(ndim):
            if n[d] > 0.0:
                hits.append(((hi[d] - pos[d]) / (n[d] * c), d, 1))
            elif n[d] < 0.0:
                hits.append(((pos[d] - lo[d]) / (-n[d] * c), d, -1))
        hits = [h for h in hits if np.isfinite(h[0]) and h[0] >= -1.0e-15]
        if not hits:
            break
        dt, d, side = min(hits, key=lambda item: item[0])
        dt = max(float(dt), 0.0)
        if t + dt > t_end:
            pos = pos + n * c * (t_end - t)
            t = t_end
            break
        pos = pos + n * c * dt
        t += dt
        pos[d] = hi[d] if side > 0 else lo[d]
        if absorb_hi[d] if side > 0 else absorb_lo[d]:
            escaped = True
        else:
            n[d] = -n[d]
    return pos, n, escaped, t


def yt_positions(data, dims):
    # Plotfile coordinates use mesh-axis names; native momentum components
    # retain physical x/y/z names in every dimensionality.
    axial_mesh_name = {1: "x", 2: "y", 3: "z"}[dims]
    z = np.asarray(data["photons", "particle_position_" + axial_mesh_name].v)
    if dims == 1:
        return np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    x = np.asarray(data["photons", "particle_position_x"].v)
    if dims == 2:
        return np.column_stack([x, np.zeros_like(z), z])
    y = np.asarray(data["photons", "particle_position_y"].v)
    return np.column_stack([x, y, z])


dims = int(args.dims)
t_end = 0.5 / c
weight = 1.0e12
if dims == 1:
    lo = np.array([0.0])
    hi = np.array([1.0])
    absorb_lo = np.array([True])
    absorb_hi = np.array([False])
    photons = [
        (np.array([0.30]), np.array([-1.0])),
        (np.array([0.70]), np.array([1.0])),
    ]
elif dims == 2:
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    absorb_lo = np.array([True, False])
    absorb_hi = np.array([True, False])
    photons = [
        (np.array([0.90, 0.50]), np.array([1.0, 0.0])),
        (np.array([0.10, 0.50]), np.array([-1.0, 0.0])),
        (np.array([0.50, 0.90]), np.array([0.0, 1.0])),
        (np.array([0.50, 0.10]), np.array([0.0, -1.0])),
        (np.array([0.70, 0.99]), np.array([0.05, 1.0])),
        (np.array([0.96, 0.99]), np.array([0.3, 1.0])),
    ]
else:
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    absorb_lo = np.array([True, True, False])
    absorb_hi = np.array([True, True, False])
    photons = [
        (np.array([0.90, 0.50, 0.50]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.10, 0.50, 0.50]), np.array([-1.0, 0.0, 0.0])),
        (np.array([0.50, 0.90, 0.50]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.50, 0.10, 0.50]), np.array([0.0, -1.0, 0.0])),
        (np.array([0.50, 0.50, 0.90]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.50, 0.50, 0.10]), np.array([0.0, 0.0, -1.0])),
        (np.array([0.70, 0.50, 0.99]), np.array([0.05, 0.0, 1.0])),
        (np.array([0.96, 0.50, 0.99]), np.array([0.3, 0.0, 1.0])),
        (np.array([0.75, 0.50, 0.50]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.50, 0.50, 0.75]), np.array([0.0, 0.0, 1.0])),
    ]

expected_escaped = 0.0
expected_retained = 0.0
expected_absorbed = 0.0
expected_positions = []
for pos, direction in photons:
    energy = input_photon_energy(weight, direction)
    final_pos, final_n, escaped, residence_time = trace(
        pos, direction, t_end, lo, hi, absorb_lo, absorb_hi
    )
    transmitted = energy * np.exp(-args.absorption_coefficient * c * residence_time)
    expected_absorbed += energy - transmitted
    if escaped:
        expected_escaped += transmitted
    else:
        expected_retained += transmitted
        expected_positions.append((final_pos, final_n))

# Some supported yt versions hold only a weak dataset reference in all_data().
initial_dataset = yt.load(args.initial)
final_dataset = yt.load("diags/plt000002")
initial = initial_dataset.all_data()
final = final_dataset.all_data()
initial_energy = yt_photon_energy(initial)
np.testing.assert_allclose(
    initial_energy,
    expected_escaped + expected_retained + expected_absorbed,
    rtol=128 * eps,
)
assert final["photons", "particle_weight"].size == len(expected_positions)
np.testing.assert_allclose(
    yt_photon_energy(final), expected_retained, rtol=128 * eps, atol=0
)

ledger = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
np.testing.assert_allclose(ledger[-1, 2], expected_retained, rtol=128 * eps, atol=0)
np.testing.assert_allclose(ledger[-1, 8], expected_escaped, rtol=128 * eps, atol=0)
np.testing.assert_allclose(
    ledger[-1, 6], expected_absorbed, rtol=128 * eps, atol=128 * eps * initial_energy
)
np.testing.assert_allclose(
    ledger[-1, 2] + ledger[-1, 6] + ledger[-1, 8],
    initial_energy,
    rtol=256 * eps,
    atol=0,
)
np.testing.assert_allclose(ledger[-1, 11], ledger[-1, 7], rtol=128 * eps, atol=1.0e-30)
np.testing.assert_allclose(ledger[-1, 13], 0.0, atol=1.0e-30)

final_pos = yt_positions(final, dims)
used = np.zeros(len(expected_positions), dtype=bool)
for pos, direction in expected_positions:
    if dims == 1:
        xyz = np.array([0.0, 0.0, pos[0]])
        mom = np.array([0.0, 0.0, direction[0]])
    elif dims == 2:
        xyz = np.array([pos[0], 0.0, pos[1]])
        mom = np.array([direction[0], 0.0, direction[1]])
    else:
        xyz = pos
        mom = direction
    distance = np.linalg.norm(final_pos - xyz, axis=1)
    match = int(np.argmin(np.where(used, np.inf, distance)))
    used[match] = True
    np.testing.assert_allclose(final_pos[match], xyz, rtol=0, atol=256 * eps)
    measured = np.array(
        [
            final["photons", "particle_momentum_x"].v[match],
            final["photons", "particle_momentum_y"].v[match],
            final["photons", "particle_momentum_z"].v[match],
        ]
    )
    measured = measured / np.linalg.norm(measured)
    np.testing.assert_allclose(
        measured, mom / np.linalg.norm(mom), rtol=256 * eps, atol=256 * eps
    )

if dims == 3:
    assert final["tracer", "particle_weight"].size == 1
    u = 0.5
    travel = 0.5 * u / np.sqrt(1.0 + u * u)
    expected_tracer = 2.0 - (0.95 + travel)
    np.testing.assert_allclose(
        final["tracer", "particle_position_x"].v,
        expected_tracer,
        rtol=128 * eps,
        atol=128 * eps,
    )
    np.testing.assert_allclose(
        final["tracer", "particle_momentum_x"].v,
        -initial["tracer", "particle_momentum_x"].v,
        rtol=128 * eps,
        atol=0,
    )

manifest_path = Path("diags/chk000002/RadiationPhotonBoundary_data.txt")
if dims == 2:
    assert not manifest_path.exists()
elif dims == 1:
    assert (
        manifest_path.read_text().strip()
        == "photon_boundary_faces_v1 photons 1 absorbing reflecting"
    )
else:
    assert (
        manifest_path.read_text().strip()
        == "photon_boundary_faces_v1 photons 3 absorbing absorbing reflecting "
        "absorbing absorbing reflecting"
    )

print(
    f"Cartesian {dims}D mixed photon faces: escaped={expected_escaped:.16e} J "
    f"retained={expected_retained:.16e} J absorbed={expected_absorbed:.16e} J"
)
