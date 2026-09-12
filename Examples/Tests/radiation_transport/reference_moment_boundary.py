#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent angular quadrature for a candidate gray M1 boundary flux.

This verifies algebra only. It neither enables a runtime boundary nor qualifies
the implicit moving-material transport operator at a physical boundary.
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def outgoing_factored(energy, flux_over_c, axis, side):
    q = np.asarray(flux_over_c, dtype=float)
    f = np.linalg.norm(q) / energy
    assert energy > 0 and 0 <= f < 1 and side in (-1, 1)
    root = np.sqrt(4 - 3 * f * f)
    b = 3 * q / (energy * (2 + root))
    delta = 12 * (1 - f) * (1 + f) / ((root + 1) * (root + 2))
    bn = side * b[axis]
    d = delta + bn * bn
    r = np.sqrt(d)
    a = -bn / r
    s = 1 + bn / r if bn >= 0 else delta / (r * (r - bn))
    denominator = 3 + np.dot(b, b)
    outgoing_energy = energy * r * s**3 * (3 + a) / (4 * denominator)
    normal_momentum = energy * d * s**3 / (2 * denominator)
    momentum = b * outgoing_energy
    momentum[axis] = side * normal_momentum
    # Fluxes divided by c, with q=F/c as the transported momentum-like state.
    return np.concatenate(([outgoing_energy], momentum))


def outgoing_quadrature(energy, flux_over_c, axis, side, order=96):
    q = np.asarray(flux_over_c, dtype=float)
    f = np.linalg.norm(q) / energy
    b = 3 * q / (energy * (2 + np.sqrt(4 - 3 * f * f)))
    points, weights = leggauss(order)
    mu = (points + 1) / 2
    weights = weights / 2
    phi = 2 * np.pi * (np.arange(256) + 0.5) / 256
    rays = np.empty((order, len(phi), 3))
    transverse = [d for d in range(3) if d != axis]
    rays[..., axis] = side * mu[:, None]
    rays[..., transverse[0]] = np.sqrt(1 - mu[:, None] ** 2) * np.cos(phi)
    rays[..., transverse[1]] = np.sqrt(1 - mu[:, None] ** 2) * np.sin(phi)
    b2 = np.dot(b, b)
    angular_energy = energy * 3 * (1 - b2) ** 3 / (4 * np.pi * (3 + b2))
    angular_energy = angular_energy / (1 - np.einsum("ijd,d->ij", rays, b)) ** 4
    transported = (
        angular_energy * mu[:, None] * weights[:, None] * (2 * np.pi / len(phi))
    )
    return np.concatenate(
        ([np.sum(transported)], np.sum(transported[..., None] * rays, axis=(0, 1)))
    )


def check_kernel(path):
    records = np.loadtxt(path)
    assert records.shape[1] == 10
    count = 0
    for record in records:
        energy, q = record[0], record[1:4]
        axis, side = int(record[4]), int(record[5])
        measured = record[6:10]
        normalized = q / energy
        f = np.linalg.norm(normalized)
        if f < 0.96:
            expected = outgoing_quadrature(1, normalized, axis, side)
            assert np.max(abs(measured - expected)) < 1e-11, record
        elif f >= 1:
            mu = side * normalized[axis]
            expected = np.zeros(4)
            if mu > 0:
                expected = np.concatenate(([mu], mu * normalized))
            assert np.max(abs(measured - expected)) < 1e-13, record
        else:
            # Axial high-f states use long-double limits, including the tiny
            # backward tail. Do not accept zero merely because it is small/E.
            assert energy == 1 and q[1] == 0 and q[2] == 0
            f = np.longdouble(q[0])
            root = np.sqrt(4 - 3 * f * f)
            b = 3 * f / (2 + root)
            delta = 12 * (1 - f) * (1 + f) / ((root + 1) * (root + 2))
            rest_energy = 3 * delta / (3 + b * b)
            expected = np.zeros(4, dtype=np.longdouble)
            if axis == 0:
                gap = delta / (1 + b)
                backwards_energy = gap**3 * (3 + b) / (4 * (3 + b * b))
                backwards_pressure = gap**3 / (2 * (3 + b * b))
                if side < 0:
                    expected[0] = backwards_energy
                    expected[1] = -backwards_pressure
                else:
                    expected[0] = f + backwards_energy
                    chi = (3 + 4 * f * f) / (5 + 2 * root)
                    expected[1] = chi - backwards_pressure
            else:
                expected[0] = rest_energy / (4 * np.sqrt(delta))
                expected[1] = b * expected[0]
                expected[axis + 1] = side * rest_energy / 6
            for actual, target in zip(measured, expected, strict=True):
                assert abs(actual - target) <= 1e-10 * abs(target), (record, expected)
        count += 1
    print(f"{count} exported CPU/GPU boundary fluxes pass independent references")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-output", type=Path)
    args = parser.parse_args()
    if args.kernel_output:
        check_kernel(args.kernel_output)
        raise SystemExit(0)
    maximum = 0.0
    count = 0
    for f in (0, 0.2, 0.7, 0.95):
        for direction in (np.array([1, 0, 0]), np.array([0.6, 0.8, 0])):
            for axis in range(3):
                for side in (-1, 1):
                    exact = outgoing_factored(1, f * direction, axis, side)
                    quadrature = outgoing_quadrature(1, f * direction, axis, side)
                    error = np.max(abs(exact - quadrature))
                    assert error < 1e-11, (f, direction, axis, side, error)
                    maximum = max(maximum, error)
                    count += 1
    print(
        f"{count} angular boundary checks; max absolute normalized error={maximum:.6e}"
    )
