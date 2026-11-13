"""Measure signed source deposition through both x faces of a closed fine patch."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--ratio", nargs=3, type=int)
parser.add_argument("--esirkepov", action="store_true")
args = parser.parse_args()
ratio = args.ratio or (1, 1, 1)
h, dt = 0.001, 1e-12
geometry.dims = 3
waveform = Path("straddling_waveform.txt")
waveform.write_text("0 1\n1e-9 1\n")
ncells = (96, 48, 48)
settings = [
    "straddling-antenna",
    "geometry.dims=3",
    "geometry.prob_lo=0 0 0",
    "geometry.prob_hi=0.096 0.048 0.048",
    "amr.n_cell=" + " ".join(str(n // r) for n, r in zip(ncells, ratio, strict=True)),
    f"amr.max_level={int(bool(args.ratio))}",
    "amr.max_grid_size=128",
    "amr.blocking_factor=2",
    "amr.n_error_buf=0",
    "boundary.field_lo=periodic periodic periodic",
    "boundary.field_hi=periodic periodic periodic",
    "boundary.particle_lo=periodic periodic periodic",
    "boundary.particle_hi=periodic periodic periodic",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    "algo.current_deposition=" + ("esirkepov" if args.esirkepov else "villasenor"),
    "algo.particle_shape=3",
    f"warpx.const_dt={dt}",
    "warpx.n_field_gather_buffer=0",
    "warpx.n_current_deposition_buffer=0",
    "warpx.do_initial_div_cleaning=0",
    "warpx.use_filter=1",
    "warpx.filter_npass_each_dir=1 1 1",
    "warpx.current_injection=1",
    f"warpx.current_injection.file={waveform}",
    "warpx.current_injection.n_pairs=3",
    "max_step=1",
    "warpx.verbose=0",
]
if args.ratio:
    settings += [
        "amr.ref_ratio_vect=" + " ".join(map(str, ratio)),
        "warpx.fine_tag_lo=0.032 0.008 0.008",
        "warpx.fine_tag_hi=0.064 0.040 0.040",
        "warpx.composite_yee_surface=1",
        "warpx.composite_yee_fixed_shape_current=1",
        "warpx.composite_yee_fixed_shape_gather=1",
    ]
for index, (xlo, xhi, sign) in enumerate(((0.028, 0.036, 1), (0.060, 0.068, -1))):
    prefix = f"warpx.current_injection.pair_{index}.drive."
    settings += [
        f"{prefix}xlo={xlo}",
        f"{prefix}xhi={xhi}",
        f"{prefix}ylo=0.020",
        f"{prefix}yhi=0.028",
        f"{prefix}zlo=0.020",
        f"{prefix}zhi=0.028",
        f"{prefix}A=0.008*0.008",
        f"{prefix}dir=0",
        f"{prefix}sign={sign}",
    ]
# The third antenna stays on the parent, close enough to the periodic face
# that a full-domain fixed-shape deposit would need to wrap its stencil.
prefix = "warpx.current_injection.pair_2.drive."
settings += [
    f"{prefix}xlo=0.000",
    f"{prefix}xhi=0.004",
    f"{prefix}ylo=0.020",
    f"{prefix}yhi=0.028",
    f"{prefix}zlo=0.020",
    f"{prefix}zhi=0.028",
    f"{prefix}A=0.004*0.008",
    f"{prefix}dir=1",
    f"{prefix}sign=1",
]

libwarpx.initialize(settings)
libwarpx.warpx.evolve(1)
fields = libwarpx.warpx.multifab_register()


def integral(field, lo, hi):
    result = 0.0
    intervals = (
        ((0, 0.0, 0.032), (1, 0.032, 0.064), (0, 0.064, 0.096))
        if args.ratio
        else ((0, 0.0, 0.096),)
    )
    for lev, xmin, xmax in intervals:
        a, b = max(lo, xmin), min(hi, xmax)
        if a >= b:
            continue
        dx, dy, dz = (h, h, h) if lev else tuple(h * r for r in ratio)
        origin = (0.032, 0.008, 0.008) if lev else (0.0, 0.0, 0.0)
        arr = np.asarray(fields.get(field, dir="x", level=lev)[...])
        ix0, ix1 = (round((x - origin[0]) / dx) for x in (a, b))
        iy0, iy1 = (round((y - origin[1]) / dy) for y in (0.010, 0.038))
        iz0, iz1 = (round((z - origin[2]) / dz) for z in (0.010, 0.038))
        result += arr[ix0:ix1, iy0:iy1, iz0:iz1].sum() * dx * dy * dz
    return result


for a, b, target in ((0.012, 0.044, 0.008), (0.050, 0.082, -0.008)):
    j = integral("current_fp", a, b)
    e = integral("Efield_fp", a, b)
    np.testing.assert_allclose(j, target, rtol=2e-9)
    np.testing.assert_allclose(e, -dt * target / 8.8541878188e-12, rtol=2e-9)

# Count every periodic parent node exactly once. The separated x antennas
# contribute zero to Jy, and the near-wall antenna has no support on the patch.
parent_cells = tuple(n // r for n, r in zip(ncells, ratio, strict=True))
for field, expected in (
    ("current_fp", 0.008),
    ("Efield_fp", -dt * 0.008 / 8.8541878188e-12),
):
    jy = np.asarray(fields.get(field, dir="y", level=0)[...])
    total = jy[tuple(slice(0, n) for n in parent_cells)].sum() * h**3 * np.prod(ratio)
    np.testing.assert_allclose(total, expected, rtol=2e-9)
libwarpx.finalize()
