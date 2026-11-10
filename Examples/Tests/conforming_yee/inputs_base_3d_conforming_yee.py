"""Transmit a vacuum pulse through a 3D conforming Yee interface."""

import argparse
from pathlib import Path

import numpy as np
from native_yee_mesh import host, mesh_fields

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--layout", choices=("box", "stepped"), required=True)
parser.add_argument("--absorber", action="store_true")
parser.add_argument("--restart", type=Path)
parser.add_argument("--invalid", choices=("gather", "current", "cleaning"))
args = parser.parse_args()

n = 32
ratio = np.array([2, 2, 1])
c = 299792458.0
length = 0.1
dt = 0.45 * length / (2 * n * c)
steps = 120
geometry.dims = 3
settings = [
    "conforming-yee",
    "geometry.dims=3",
    "geometry.prob_lo=-0.05 -0.05 -0.05",
    "geometry.prob_hi=0.05 0.05 0.05",
    f"amr.n_cell={n} {n} {n}",
    "amr.max_level=1",
    "amr.max_grid_size=16",
    "amr.blocking_factor=4",
    "amr.n_error_buf=0",
    "amr.grid_eff=1",
    "amr.ref_ratio_vect=2 2 1",
    "warpx.composite_yee_surface=1",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    "warpx.use_filter=0",
    "warpx.do_initial_div_cleaning=0",
    "warpx.do_dive_cleaning=0",
    "warpx.do_divb_cleaning=0",
    "warpx.do_pml_dive_cleaning=0",
    "warpx.do_pml_divb_cleaning=0",
    f"warpx.const_dt={dt:.17g}",
    f"max_step={steps}",
    "warpx.verbose=0",
    "amrex.the_arena_init_size=0",
]
if args.layout == "stepped":
    settings.append(
        "warpx.ref_patch_function(x,y,z)=if((x*x+y*y<0.021^2)*(abs(z)<0.019),1,0)"
    )
else:
    settings.extend(
        ["warpx.fine_tag_lo=-0.02 -0.02 -0.02", "warpx.fine_tag_hi=0.02 0.02 0.02"]
    )
boundary = "pml periodic periodic" if args.absorber else "periodic periodic periodic"
settings.extend([f"boundary.field_lo={boundary}", f"boundary.field_hi={boundary}"])
if args.absorber:
    settings.extend(
        [
            "warpx.pml_ncell=4",
            "warpx.pml_ncell_vect=4 3 2",
            "warpx.pml_delta_vect=4 3 2",
            "warpx.pml_delta=4",
            "diagnostics.diags_names=chk",
            "chk.diag_type=Full",
            "chk.format=checkpoint",
            "chk.file_prefix=diags/checkpoint",
            "chk.intervals=60,120",
        ]
    )
if args.restart:
    settings.append(f"amr.restart={args.restart}/diags/checkpoint000060")
if args.invalid == "gather":
    settings.append("warpx.composite_yee_fixed_shape_gather=1")
elif args.invalid == "current":
    settings.extend(
        ["warpx.composite_yee_surface=0", "warpx.composite_yee_fixed_shape_current=1"]
    )
elif args.invalid == "cleaning":
    settings.append("warpx.do_dive_cleaning=1")
libwarpx.initialize(settings)
fields = libwarpx.warpx.multifab_register()


blocks, boxes, covered = mesh_fields(fields, n, ratio)
if args.layout == "stepped":
    lower = np.min(np.argwhere(covered), axis=0)
    upper = np.max(np.argwhere(covered), axis=0)
    assert np.count_nonzero(covered) < np.prod(upper - lower + 1)

if not args.restart:
    # A divergence-free transverse electric pulse outside the fine patch.
    for level in range(2):
        h = length / (n * (ratio if level else np.ones(3)))
        for lo, hi, ey in blocks[level, "E", 1]:
            x = -length / 2 + np.arange(lo[0], hi[0] + 1)[:, None, None] * h[0]
            z = -length / 2 + np.arange(lo[2], hi[2] + 1)[None, None, :] * h[2]
            r = (x + 0.038) / 0.008
            ey[...] = np.where(abs(r) < 1, np.cos(np.pi * r / 2) ** 4, 0) * np.cos(
                2 * np.pi * z / length
            )


def cell_average(values, shape):
    for d in range(3):
        if values.shape[d] != shape[d]:
            a = np.moveaxis(values, d, 0)
            values = np.moveaxis(0.5 * (a[:-1] + a[1:]), 0, d)
    return values


def measure():
    energy = 0.0
    peak = 0.0
    for level in range(2):
        h = length / (n * (ratio if level else np.ones(3)))
        for i, (lo, hi) in enumerate(boxes[level]):
            shape = hi - lo + 1
            region = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
            active = np.ones(shape, dtype=bool) if level else ~covered[region]
            electric = [host(blocks[level, "E", d][i][2]) for d in range(3)]
            density = np.zeros(shape)
            for d in range(3):
                magnetic = host(blocks[level, "B", d][i][2])
                u, v = (d + 1) % 3, (d + 2) % 3
                curl = (
                    np.diff(electric[v], axis=u) / h[u]
                    - np.diff(electric[u], axis=v) / h[v]
                )
                # The invariant for the split Yee advance uses B(-dt/2)·B(+dt/2).
                density += cell_average(electric[d] ** 2, shape)
                density += c**2 * cell_average(
                    magnetic**2 - (0.5 * dt * curl) ** 2, shape
                )
                if level:
                    peak = max(peak, np.max(abs(electric[d])))
            energy += np.prod(h) * np.sum(density[active])
    return energy, peak


initial, _ = measure()
assert initial > 0
start = libwarpx.warpx.getistep(0)
peak = 0.0
for step in range(start, steps, 30):
    libwarpx.warpx.evolve(30)
    energy, peak_fine = measure()
    peak = max(peak, peak_fine)
    if not args.absorber:
        np.testing.assert_allclose(energy, initial, rtol=2e-12, atol=0)
assert peak > 0.01, "The pulse never reached the refined grid"
if args.absorber:
    chk = Path(f"diags/checkpoint{steps:06d}")
    assert (chk / "Level_0/pml_Bx_fp_H").is_file()
    assert not (chk / "Level_1/pml_Bx_fp_H").exists()
    assert not (chk / "Level_1/pml_Bx_cp_H").exists()
    arrays = {
        f"{level}_{family}_{d}_{i}": host(view)
        for (level, family, d), group in blocks.items()
        for i, (_, _, view) in enumerate(group)
    }
    if args.restart:
        with np.load(args.restart / "solution.npz") as reference:
            assert set(reference.files) == set(arrays)
            for key, values in arrays.items():
                np.testing.assert_allclose(
                    values,
                    reference[key],
                    atol=2e-13 * np.max(abs(reference[key])),
                    rtol=2e-13,
                )
    else:
        np.savez("solution.npz", **arrays)
libwarpx.finalize()
