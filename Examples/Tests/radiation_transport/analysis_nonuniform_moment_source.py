#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent discrete gray four-force and native nodal caloric checks.

This checks a nonuniform radiation/material stage, optionally with spatial flux.
No PIC drift or material density advance occurs during this operator test.
No WarpX source, frame, work or EOS helper is imported by this oracle.
"""

import argparse
from pathlib import Path

import numpy as np


def pressure_tensor(radiation):
    energy, q = radiation[0], radiation[1:4]
    assert energy > 0
    f2 = np.dot(q, q) / energy**2
    assert 0 <= f2 <= 1
    chi = (3 + 4 * f2) / (5 + 2 * np.sqrt(4 - 3 * f2))
    pressure = energy * (1 - chi) / 2 * np.eye(3, dtype=np.longdouble)
    if f2 > 0:
        pressure += (3 * chi - 1) / (2 * energy * f2) * np.outer(q, q)
    tensor = np.zeros((4, 4), dtype=np.longdouble)
    tensor[0, 0], tensor[0, 1:], tensor[1:, 0] = energy, q, q
    tensor[1:, 1:] = pressure
    return tensor


def transport_increment(state, beta, nx, ny, spacing, rate, dt, c):
    """Independent projected LLF flux, including the transverse tensor term."""
    dimensions = 2 if ny > 1 else 1
    axes = (0, 2) if dimensions == 2 else (2,)
    opacity = rate / (c * dt)
    tensors = [pressure_tensor(row) for row in state]
    fluxes = np.zeros((dimensions, nx * ny, 4), dtype=np.longdouble)

    def neighbor(cell, direction, shift):
        i, j = cell % nx, cell // nx
        return (
            ((i + shift) % nx) + nx * j
            if direction == 0
            else i + nx * ((j + shift) % ny)
        )

    for direction, normal in enumerate(axes):
        for right in range(nx * ny):
            left = neighbor(right, direction, -1)
            b = (beta[left] + beta[right]) / 2
            gamma = 1 / np.sqrt(1 - b @ b)
            tau = 3 * opacity * gamma * spacing[direction] / (2 * (1 - b[normal] ** 2))
            alpha, blend = 1 / (1 + tau), tau / (1 + tau)

            def projected(cell):
                return state[cell, 0] - b @ state[cell, 1:4]

            rl, rr = projected(left), projected(right)
            sl = state[left, normal + 1] - tensors[left][normal + 1, 1:] @ b
            sr = state[right, normal + 1] - tensors[right][normal + 1, 1:] @ b
            momentum = c * (
                (tensors[left][1:, normal + 1] + tensors[right][1:, normal + 1]) / 2
                - alpha * (state[right, 1:4] - state[left, 1:4]) / 2
            )
            slow = c * (
                b[normal] * (rl + rr) / 2
                - blend * abs(b[normal]) * (rr - rl) / 2
                + alpha * ((sl + sr) / 2 - b[normal] * (rl + rr) / 2 - (rr - rl) / 2)
            )
            if opacity > 0:
                for tangent, physical_axis in enumerate(axes):
                    if tangent == direction:
                        continue
                    gradient = (
                        projected(neighbor(left, tangent, 1))
                        - projected(neighbor(left, tangent, -1))
                        + projected(neighbor(right, tangent, 1))
                        - projected(neighbor(right, tangent, -1))
                    ) / (4 * spacing[tangent])
                    slow += (
                        c
                        * blend**2
                        * b[normal]
                        * b[physical_axis]
                        * gradient
                        / (3 * opacity * gamma)
                    )
            fluxes[direction, right, 0] = slow + b @ momentum
            fluxes[direction, right, 1:] = momentum
    change = np.zeros((nx * ny, 4), dtype=np.longdouble)
    for direction in range(dimensions):
        for cell in range(nx * ny):
            upper = neighbor(cell, direction, 1)
            change[cell] += (
                dt
                * (fluxes[direction, upper] - fluxes[direction, cell])
                / spacing[direction]
            )
    assert np.max(np.abs(np.sum(change, axis=0))) < 1e-14 * np.sum(state[:, 0])
    return change


def shaped_particles(path, nx, ny, spacing, c):
    """Reconstruct nodal deposition/cell averaging from raw particle records."""

    def read(name):
        data = np.loadtxt(Path(path).parent / name, dtype=np.longdouble)
        assert data.shape == (4 * nx * ny, 11)
        assert np.isfinite(data).all()
        return data[np.lexsort((data[:, 3], data[:, 2], data[:, 1]))]

    old, new = read("particles_before.txt"), read("particles_after.txt")
    assert np.array_equal(old[:, :4], new[:, :4])
    shape = np.zeros((nx * ny, len(old)), dtype=np.longdouble)
    dimensions = 2 if ny > 1 else 1
    lengths = (nx, ny)
    # Test domains start at zero. Explicitly visit charge nodes then all cells
    # sharing each node: independent of the production three-point formula.
    for p, row in enumerate(old):
        coordinates = row[1 : 1 + dimensions] / spacing[:dimensions]
        base = np.floor(coordinates).astype(int)
        fraction = coordinates - base
        for node_y in range(2 if dimensions == 2 else 1):
            for node_x in range(2):
                node = (node_x, node_y)
                weight = np.longdouble(1)
                for d in range(dimensions):
                    weight *= fraction[d] if node[d] else 1 - fraction[d]
                for corner_y in range(2 if dimensions == 2 else 1):
                    for corner_x in range(2):
                        corner = (corner_x, corner_y)
                        cell = [
                            (base[d] + node[d] - corner[d]) % lengths[d]
                            for d in range(dimensions)
                        ]
                        flat = cell[0] + (nx * cell[1] if dimensions == 2 else 0)
                        shape[flat, p] += weight / 2**dimensions
    assert np.max(abs(shape.sum(axis=0) - 1)) < 1e-15
    mass = shape @ old[:, 0]
    assert np.all(mass > 0)
    actual_du = new[:, 4:7] - old[:, 4:7]
    requested_du = actual_du + new[:, 7:10] - old[:, 7:10]
    trial = old[:, 4:7] + requested_du
    gamma0 = np.sqrt(1 + np.sum(old[:, 4:7] ** 2, axis=1) / c**2)
    gamma1 = np.sqrt(1 + np.sum(trial**2, axis=1) / c**2)
    secant = (old[:, 4:7] + trial) / (gamma0 + gamma1)[:, None]
    beta = (shape @ (old[:, 0, None] * secant)) / (mass[:, None] * c)
    return beta, shape, mass, requested_du


def check(path, shaped=False):
    with open(path) as stream:
        metadata = list(map(np.longdouble, stream.readline().split()))
        assert len(metadata) in (8, 12)
        nx, ny, volume, rate, c, kb, qe, latent = metadata[:8]
        spatial = len(metadata) == 12 and bool(metadata[9])
        data = np.loadtxt(stream, dtype=np.longdouble)
    nx, ny = int(nx), int(ny)
    assert data.shape == (nx * ny, 30)
    assert np.isfinite(data).all()
    old, new = data[:, :15], data[:, 15:]
    assert np.array_equal(old[:, 5], new[:, 5])
    assert np.array_equal(old[:, 14], new[:, 14])
    assert np.all(old[:, 14] == 4)
    assert np.ptp(old[:, 4]) > np.longdouble("0.1") * np.mean(old[:, 4])
    u0 = old[:, 6:9] / old[:, 5, None]
    u1 = new[:, 6:9] / new[:, 5, None]
    u0[:, 2] += np.longdouble("1e5")
    u1[:, 2] += np.longdouble("1e5")
    gamma0 = np.sqrt(1 + np.sum(u0**2, axis=1) / c**2)
    gamma1 = np.sqrt(1 + np.sum(u1**2, axis=1) / c**2)
    beta = (u0 + u1) / ((gamma0 + gamma1)[:, None] * c)
    if shaped:
        assert len(metadata) == 12
        beta, shape, receiver_mass, requested_du = shaped_particles(
            path, nx, ny, metadata[10:12], c
        )
    transported = np.zeros((nx * ny, 4), dtype=np.longdouble)
    if spatial:
        transported = transport_increment(
            new, beta, nx, ny, metadata[10:12], rate, metadata[8], c
        )
    thermal = (
        old[:, 0] - new[:, 0] - (new[:, 9] - old[:, 9]) - (new[:, 13] - old[:, 13])
    ) - transported[:, 0]
    momentum = c * (new[:, 6:9] - old[:, 6:9] + new[:, 10:13] - old[:, 10:13])
    radiation_change = old[:, :4] - new[:, :4] - transported
    # Momentum is checked on actual particles plus their signed numerical carry.
    if shaped:
        # Pointwise deposited particle change is not the source impulse when
        # clouds overlap. Check every actual particle against its gathered
        # source instead; the independent global inventory is checked too.
        predicted_du = shape.T @ (
            radiation_change[:, 1:] / (c * receiver_mass[:, None])
        )
        momentum_error = np.max(
            abs(requested_du - predicted_du)
            / np.maximum(abs(predicted_du), np.longdouble("1e-8"))
        )
        assert np.max(
            abs(np.sum(momentum - radiation_change[:, 1:], axis=0))
        ) < 1e-10 * np.sum(old[:, 0])
        source_work = np.sum(beta * radiation_change[:, 1:], axis=1)
        thermal = radiation_change[:, 0] - source_work
    else:
        momentum_error = np.max(
            np.abs(momentum - radiation_change[:, 1:])
            / np.maximum(
                np.abs(radiation_change[:, 1:]), old[:, 0, None] * np.longdouble("1e-8")
            )
        )
    assert momentum_error < 1e-10, momentum_error

    def corners(cell):
        i, j = cell % nx, cell // nx
        return [
            ((i + di) % nx) + nx * ((j + dj) % ny)
            for dj in range(2 if ny > 1 else 1)
            for di in range(2)
        ]

    caloric_error = np.zeros(nx * ny, dtype=np.longdouble)
    old_ev, new_ev = old[:, 4] * kb / qe, new[:, 4] * kb / qe
    # Density is homogeneous; the source test deliberately keeps kinetic ions
    # fixed in position during this one local operator application.
    number = np.longdouble("1e20") * volume
    old_cv = np.full(nx * ny, np.longdouble("1.5"))
    old_u = np.longdouble("1.5") * old_ev
    new_u = np.longdouble("1.5") * new_ev
    if latent:
        old_u += 2 * old_ev**4 / (1 + old_ev**4)
        new_u += 2 * new_ev**4 / (1 + new_ev**4)
        old_cv += 8 * old_ev**3 / (1 + old_ev**4) ** 2
    for cell in range(nx * ny):
        nodes = corners(cell)
        caloric_error[nodes] += thermal[cell] * old_cv[nodes] / np.sum(old_cv[nodes])
    actual_caloric = number * qe * (new_u - old_u)
    node_error = np.max(np.abs(actual_caloric - caloric_error)) / np.max(
        np.abs(caloric_error)
    )
    assert node_error < 1e-10, node_error
    total_error = abs(
        np.sum(radiation_change[:, 0])
        - np.sum(actual_caloric)
        - np.sum(new[:, 9] - old[:, 9])
        - np.sum(new[:, 13] - old[:, 13])
    )
    total_error /= np.sum(old[:, 0])
    assert total_error < 1e-10, total_error

    worst_source = np.longdouble(0)
    for cell in range(nx * ny):
        tensor = pressure_tensor(new[cell])
        b = beta[cell]
        b2 = np.dot(b, b)
        gamma = 1 / np.sqrt(1 - b2)
        boost = np.eye(4, dtype=np.longdouble)
        boost[0, 0] = gamma
        boost[0, 1:] = boost[1:, 0] = -gamma * b
        boost[1:, 1:] += gamma**2 / (gamma + 1) * np.outer(b, b)
        rest = boost @ tensor @ boost.T
        bath = (
            np.longdouble("7.565733250280007e-16")
            * volume
            * np.mean(new[corners(cell), 4] ** 4)
        )
        force_rest = rate * rest[0].copy()
        force_rest[0] -= rate * bath
        boost[0, 1:] *= -1
        boost[1:, 0] *= -1
        force = boost @ force_rest
        scale = np.maximum(np.abs(force), old[cell, 0] * np.longdouble("1e-8"))
        worst_source = max(
            worst_source, np.max(np.abs(radiation_change[cell] - force) / scale)
        )
    assert worst_source < 1e-10, worst_source
    assert np.max(np.abs(new[:, 4] - old[:, 4]) / old[:, 4]) > 0.01
    print(
        f"Nonuniform {'spatial' if spatial else 'source'}: four-force={worst_source:.3e}, "
        f"nodal caloric={node_error:.3e}, "
        f"energy={total_error:.3e}, particle momentum={momentum_error:.3e}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state")
    parser.add_argument("--shaped", action="store_true")
    args = parser.parse_args()
    check(args.state, args.shaped)
