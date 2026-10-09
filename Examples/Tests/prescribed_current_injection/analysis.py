#!/usr/bin/env python3
"""Quantitative checks for prescribed-current deposition."""

import argparse
import glob
import math

import numpy as np
import yt
from scipy.constants import epsilon_0

yt.funcs.mylog.setLevel(50)


def get_field(ad, name):
    matches = [field for field in ad.ds.field_list if field[1] == name]
    if len(matches) != 1:
        raise AssertionError(f"Expected one field named {name!r}, found {matches}")
    return np.asarray(ad[matches[0]])


def check_extremum(ad, name, expected, rtol=2.0e-6):
    values = get_field(ad, name)
    actual = values.max() if expected >= 0.0 else values.min()
    error = abs(actual - expected) / max(abs(expected), 1.0)
    print(
        f"{name}: expected={expected:.12e}, actual={actual:.12e}, rel_error={error:.3e}"
    )
    assert error < rtol


def check_integral(ds, name, expected, rtol=2.0e-6):
    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    values = get_field(grid, name)
    cell_size = (ds.domain_right_edge - ds.domain_left_edge) / ds.domain_dimensions
    cell_measure = float(np.prod(cell_size[: ds.dimensionality]))
    actual = np.sum(values) * cell_measure
    error = abs(actual - expected) / max(abs(expected), 1.0)
    print(
        f"integral({name}): expected={expected:.12e}, "
        f"actual={actual:.12e}, rel_error={error:.3e}"
    )
    assert error < rtol


def check_zero(ad, name, atol=1.0e-20):
    maximum = np.max(np.abs(get_field(ad, name)))
    print(f"{name}: max_abs={maximum:.12e}")
    assert maximum < atol


def check_gauss_law(ad, rtol=5.0e-2):
    rho = get_field(ad, "rho")
    div_e = get_field(ad, "divE")
    charge_scale = np.max(np.abs(rho / epsilon_0))
    assert charge_scale > 0.0, "Finite antenna must deposit polarization charge"
    error = np.max(np.abs(div_e - rho / epsilon_0)) / charge_scale
    print(f"Gauss-law relative L-infinity error={error:.6e}")
    assert error < rtol


def check_continuity(plotfiles, rtol=2.0e-6):
    """Check d(rho)/dt + div(J) on cell-centered plotfile fields.

    For a periodic Yee mesh, averaging the native staggered rho and J fields
    to cell centers commutes with the centered divergence used below.  This is
    therefore an independent regression of the source's old/new charge and
    charge-conserving current deposition, rather than another Gauss-law test.
    """
    assert len(plotfiles) >= 2, "Continuity check needs consecutive plotfiles"
    for old_path, new_path in zip(plotfiles[:-1], plotfiles[1:]):
        ds_old = yt.load(old_path)
        ds_new = yt.load(new_path)
        old = ds_old.covering_grid(
            level=0,
            left_edge=ds_old.domain_left_edge,
            dims=ds_old.domain_dimensions,
        )
        new = ds_new.covering_grid(
            level=0,
            left_edge=ds_new.domain_left_edge,
            dims=ds_new.domain_dimensions,
        )

        dt = float(ds_new.current_time - ds_old.current_time)
        assert dt > 0.0
        rho_rate = (get_field(new, "rho") - get_field(old, "rho")) / dt
        spacing = np.asarray(
            (ds_new.domain_right_edge - ds_new.domain_left_edge)
            / ds_new.domain_dimensions
        )
        div_j = np.zeros_like(rho_rate)
        for axis, name in enumerate(("jx", "jy", "jz")):
            current = get_field(new, name)
            div_j += (
                np.roll(current, -1, axis=axis) - np.roll(current, 1, axis=axis)
            ) / (2.0 * spacing[axis])

        scale = max(np.max(np.abs(rho_rate)), np.max(np.abs(div_j)))
        assert scale > 0.0, "Finite antenna must have a nonzero continuity scale"
        error = np.max(np.abs(rho_rate + div_j)) / scale
        print(
            f"continuity {old_path} -> {new_path}: "
            f"relative L-infinity error={error:.6e}"
        )
        assert error < rtol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jx-integral", type=float, default=0.0)
    parser.add_argument("--jy-integral", type=float, default=0.0)
    parser.add_argument("--jz-integral", type=float, default=0.0)
    parser.add_argument("--jz", type=float)
    parser.add_argument("--rz-current", type=float)
    parser.add_argument("--check-continuity", action="store_true")
    args = parser.parse_args()

    plotfiles = sorted(
        path for path in glob.glob("diags/diag1*") if ".old." not in path
    )
    assert plotfiles, "No prescribed-current plotfile found"
    ds = yt.load(plotfiles[-1])
    ad = ds.all_data()

    if args.check_continuity:
        check_continuity(plotfiles)
    check_gauss_law(ad)
    if args.rz_current is None:
        for name, expected in (
            ("jx", args.jx_integral),
            ("jy", args.jy_integral),
            ("jz", args.jz_integral),
        ):
            if expected == 0.0:
                check_zero(ad, name)
            else:
                check_integral(ds, name, expected)
        return

    check_zero(ad, "jt")
    assert args.jz is not None
    check_extremum(ad, "jz", args.jz)

    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    jr = np.squeeze(get_field(grid, "jr"))
    nr, nz = (int(v) for v in ds.domain_dimensions[:2])
    assert jr.shape == (nr, nz), (jr.shape, nr, nz)
    dr = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]) / nr)
    dz = float((ds.domain_right_edge[1] - ds.domain_left_edge[1]) / nz)
    radius = float(ds.domain_left_edge[0]) + (np.arange(nr) + 0.5) * dr
    shell_current = 2.0 * math.pi * radius * dz * np.sum(jr, axis=1)
    actual = np.max(np.abs(shell_current))
    error = abs(actual - args.rz_current) / abs(args.rz_current)
    print(
        f"RZ shell current: expected={args.rz_current:.12e}, "
        f"actual={actual:.12e}, rel_error={error:.3e}"
    )
    assert error < 2.0e-2


if __name__ == "__main__":
    main()
