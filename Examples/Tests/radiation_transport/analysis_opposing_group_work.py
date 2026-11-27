#!/usr/bin/env python3
"""Independent 1D FLD face fluxes and symmetric spectral kick/work oracle."""

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action="store_true")
args = parser.parse_args()


def load(path):
    with (path / "Header").open() as header:
        header.readline()
        names = [header.readline().strip() for _ in range(int(header.readline()))]
    return _read_buffer(str(path), str(path / "Level_0/Cell_H"), names)


names = [f"radiation_diffusion_energy_g{g}" for g in range(2)]
if args.restart:
    reference = load(
        Path("../test_1d_radiation_transport_opposing_group_work/diags/diag000002")
    )
    actual = load(Path("diags/diag000002"))
    for name in names:
        np.testing.assert_allclose(actual[name], reference[name], rtol=3e-12, atol=0)
else:
    c = 299792458.0
    dx, dt, sigma = 1 / 16, 1e-8, 4e6
    mass = 1e20 * 1e-27 * dx
    z = (np.arange(16) + 0.5) * dx
    energy = np.stack((1e6 + 1e5 * z, 1e6 - 1e5 * z)) * dx
    velocity = np.full(16, 1e6)
    initial = load(Path("diags/diag000000"))
    for group, name in enumerate(names):
        np.testing.assert_allclose(np.squeeze(initial[name]), energy[group], rtol=3e-12)
    for step in (1, 2):
        density = energy / dx
        gradient = np.diff(density, axis=1) / dx
        face_density = 0.5 * (density[:, :-1] + density[:, 1:])
        r = np.abs(gradient) / (sigma * face_density)
        limiter = (2 + r) / (6 + 3 * r + r * r)
        flux = np.zeros((2, 17))  # Reflecting exterior faces.
        flux[:, 1:-1] = -c * limiter * gradient / sigma
        impulse = dt * dx * sigma / c * 0.5 * (flux[:, :-1] + flux[:, 1:])
        energy -= dt * np.diff(flux, axis=1)
        group_work = np.zeros_like(energy)
        for group in (0, 1, 1, 0):
            old = velocity.copy()
            velocity += 0.5 * impulse[group] / mass
            delta = velocity - old
            gamma_old = np.sqrt(1 + (old / c) ** 2)
            gamma_new = np.sqrt(1 + (velocity / c) ** 2)
            work = mass * (2 * old * delta + delta * delta) / (gamma_old + gamma_new)
            energy[group] -= work
            group_work[group] += work
        # Net force cancellation must not imply zero work for each group.
        assert group_work[0].sum() < -100
        assert group_work[1].sum() > 100
        actual = load(Path(f"diags/diag{step:06d}"))
        for group, name in enumerate(names):
            np.testing.assert_allclose(
                np.squeeze(actual[name]), energy[group], rtol=3e-12, atol=0
            )
        np.testing.assert_allclose(
            np.squeeze(actual["radiation_material_kinetic_energy"]),
            group_work.sum(axis=0),
            rtol=3e-10,
            atol=1e-8,
        )
        print(f"step {step}: signed group work {group_work.sum(axis=1)} J")
print("Opposing-group work/restart passed")
