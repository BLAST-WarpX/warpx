#!/usr/bin/env python3
"""Validate SMR material coordinates, composite current, and field constraints."""

import argparse
from pathlib import Path

import numpy as np
import yt
from scipy.constants import epsilon_0, mu_0

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=("port", "antenna", "material"), required=True)
parser.add_argument("--dims", choices=("2", "3", "RZ"), required=True)
parser.add_argument("--method", default="backwardeuler")
parser.add_argument("--reference")
parser.add_argument("--eb", action="store_true")
parser.add_argument("--implicit", action="store_true")
parser.add_argument("--theta", type=float, default=0.5)
args = parser.parse_args()
yt.funcs.mylog.setLevel(50)
plotfiles = sorted(
    path for path in Path("diags").glob("diag1*") if path.name[5:].isdigit()
)
assert plotfiles
ds = yt.load(str(plotfiles[-1]))
assert ds.index.max_level == 1
data = ds.all_data()


def field(name):
    return np.asarray(data[next(item for item in ds.field_list if item[1] == name)])


x = np.asarray(data["index", "x"])
z = np.asarray(data["index", "z" if args.dims == "3" else "y"])
for name, expected in (
    ("sigma", epsilon_0 * 1.0e10 * (2 + x)),
    ("epsilon", epsilon_0 * (2 + x)),
    ("mu", mu_0 * (2 + 0.5 * z)),
):
    np.testing.assert_allclose(field(name), expected, rtol=5.0e-6, atol=0)
    print(f"{name}: correct parser coordinates on both levels")

if args.source == "antenna":
    # Fine current diagnostics contain patch-local sources. The synchronized
    # level-0 current contains every level's source, including interface guards.
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    name = "jz"
    current = np.squeeze(
        np.asarray(grid[next(item for item in ds.field_list if item[1] == name)])
    )
    ndim = 3 if args.dims == "3" else 2
    dx = np.asarray(
        (ds.domain_right_edge - ds.domain_left_edge) / ds.domain_dimensions
    )[:ndim]
    volume = np.prod(dx)
    if args.dims == "RZ":
        radii = (
            float(ds.domain_left_edge[0]) + (np.arange(current.shape[0]) + 0.5) * dx[0]
        )
        volume = 2 * np.pi * radii[:, None] * volume
        expected = 750.0
    else:
        expected = 750.0 if args.dims == "3" else 1500.0
    integral = np.sum(current * volume)
    print(f"synchronized integral of {name}: {integral}, expected {expected}")
    np.testing.assert_allclose(integral, expected, rtol=2.0e-6, atol=0)
    assert np.max(np.abs(field("rho"))) > 0
elif args.source == "material":
    # sigma/epsilon is spatially constant; uniform Ey has zero curl and decays
    # by the chosen conductivity update. Check cells away from patch PMLs.
    decay = 1 / 1.01 if args.method == "backwardeuler" else (1 - 0.005) / (1 + 0.005)
    if args.implicit:
        decay = (1 - (1 - args.theta) * 0.01) / (1 + args.theta * 0.01)
    mask = ((x > 0.125) & (x < 0.3) if args.dims == "RZ" else np.abs(x) < 0.2) & (
        np.abs(z) < 0.2
    )
    if args.dims == "3":
        mask &= np.abs(np.asarray(data["index", "y"])) < 0.2
    if args.eb:
        radius_squared = x**2
        if args.dims == "3":
            radius_squared += np.asarray(data["index", "y"]) ** 2
        mask &= radius_squared > 0.125**2
    np.testing.assert_allclose(
        field("Ez" if args.dims == "RZ" else "Ey")[mask], decay**2, rtol=2.0e-12, atol=0
    )
    print(f"refined material damping agrees with {args.method}: {decay**2}")
elif args.dims == "3":
    scale = max(np.max(np.abs(field(name))) for name in ("Bx", "By", "Bz"))
    spacing = 1 / 64
    error = np.max(np.abs(field("divB"))) / (scale / spacing)
    print(f"composite relative div(B): {error}")
    assert error < 2.0e-12

if args.reference:
    reference_dataset = yt.load(args.reference)
    reference = reference_dataset.all_data()
    for name in (
        "Er" if args.dims == "RZ" else "Ex",
        "Et" if args.dims == "RZ" else "Ey",
        "Ez",
        "Br" if args.dims == "RZ" else "Bx",
        "Bt" if args.dims == "RZ" else "By",
        "Bz",
        "rho",
        "jz",
        "sigma",
        "epsilon",
        "mu",
    ):
        expected = np.asarray(
            reference[next(item for item in reference.ds.field_list if item[1] == name)]
        )
        actual = field(name)
        scale = max(np.max(np.abs(expected)), 1.0e-30)
        error = np.max(np.abs(actual - expected)) / scale
        print(f"restart relative {name} error: {error}")
        assert error < 2.0e-10
