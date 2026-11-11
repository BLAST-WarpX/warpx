"""Measure two waveform sources and an independent proton on native XZ currents."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--slab", nargs=2, type=int, metavar=("RX", "RZ"))
parser.add_argument("--no-plasma", action="store_true")
args = parser.parse_args()

rx, rz = args.slab or (1, 1)
h, dt = 0.001, 1.0e-12
nx, nz = 96, 32
geometry.dims = 2
waveform = Path("antenna_waveform.txt")
waveform.write_text("0 1\n1e-9 1\n")
settings = [
    "xz-antenna",
    "geometry.dims=2",
    "geometry.prob_lo=0 0",
    "geometry.prob_hi=0.096 0.032",
    f"amr.n_cell={nx // rx} {nz // rz}",
    f"amr.max_level={int(bool(args.slab))}",
    "amr.max_grid_size=128",
    "amr.blocking_factor=2",
    "amr.n_error_buf=0",
    "boundary.field_lo=periodic periodic",
    "boundary.field_hi=periodic periodic",
    "boundary.particle_lo=periodic periodic",
    "boundary.particle_hi=periodic periodic",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    "algo.current_deposition=esirkepov",
    "algo.particle_shape=3",
    f"warpx.const_dt={dt}",
    "warpx.n_field_gather_buffer=0",
    "warpx.n_current_deposition_buffer=0",
    "warpx.do_initial_div_cleaning=0",
    "warpx.use_filter=1",
    "warpx.filter_npass_each_dir=1 1",
    "warpx.current_injection=1",
    f"warpx.current_injection.file={waveform}",
    "warpx.current_injection.n_pairs=2",
    "max_step=1",
    "warpx.verbose=0",
]
if args.slab:
    settings += [
        f"amr.ref_ratio_vect={rx} {rz}",
        "warpx.fine_tag_lo=0.032 0",
        "warpx.fine_tag_hi=0.064 0.032",
        "warpx.composite_xz_interface=1",
        "warpx.composite_xz_fixed_shape=1",
    ]
if not args.no_plasma:
    settings += [
        "particles.species_names=ions",
        "ions.species_type=proton",
        "ions.injection_style=singleparticle",
        "ions.single_particle_pos=0.055 0 0.022",
        "ions.single_particle_u=0 0.02 0",
        "ions.single_particle_weight=1e9",
    ]
# Each box drives one ampere per metre across an 8 mm transverse aperture.
# The two sources have opposite signs and occupy different active meshes.
for index, (xlo, xhi, sign) in enumerate(((0.010, 0.018, 1), (0.044, 0.052, -1))):
    prefix = f"warpx.current_injection.pair_{index}.drive."
    settings += [
        f"{prefix}xlo={xlo}",
        f"{prefix}xhi={xhi}",
        f"{prefix}ylo=-0.5",
        f"{prefix}yhi=0.5",
        f"{prefix}zlo=0.008",
        f"{prefix}zhi=0.016",
        f"{prefix}A=0.008",
        f"{prefix}dir=0",
        f"{prefix}sign={sign}",
    ]

libwarpx.initialize(settings)
libwarpx.warpx.evolve(1)
fields = libwarpx.warpx.multifab_register()


def integral(family, direction, level, lower, upper):
    # Include each periodic z node once and use the physical spacing of this leaf.
    dx, dz = (h, h) if level else (rx * h, rz * h)
    origin = 0.032 if level else 0.0
    values = np.squeeze(np.asarray(fields.get(family, dir=direction, level=level)[...]))
    rows = slice(round((lower - origin) / dx), round((upper - origin) / dx))
    return values[rows, : round(0.032 / dz)].sum() * dx * dz


active = int(bool(args.slab))
proton_current = 1.602176634e-19 * 1e9 * 299792458.0 * 0.02 / np.sqrt(1 + 0.02**2)
for direction, level, lower, upper, expected in (
    ("x", 0, 0.002, 0.028, 0.008),
    ("x", active, 0.034, 0.062, -0.008),
    ("y", active, 0.034, 0.062, 0.0 if args.no_plasma else proton_current),
):
    current = integral("current_fp", direction, level, lower, upper)
    np.testing.assert_allclose(current, expected, rtol=2e-9, atol=1e-14)
    electric = integral("Efield_fp", direction, level, lower, upper)
    np.testing.assert_allclose(
        electric, -dt * expected / 8.8541878188e-12, rtol=2e-9, atol=1e-14
    )
libwarpx.finalize()

