"""Integrate three waveform currents in the cylindrical Yee dual volumes at a radial join."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--uniform", action="store_true")
parser.add_argument("--plasma", action="store_true")
parser.add_argument("--parent-coil", action="store_true")
parser.add_argument("--parent-particle", action="store_true")
parser.add_argument("--guarded-seed", action="store_true")
parser.add_argument("--restart", type=Path)
args = parser.parse_args()
geometry.dims = "RZ"
h, dz, dt = 0.001, 0.001, 1e-12
nr, nz = 64, 32
radius = nr * h
profile = Path("radial_waveform.txt")
profile.write_text("0 1\n1e-9 1\n")
settings = [
    "radial-antenna",
    "geometry.dims=RZ",
    "geometry.prob_lo=0 0",
    "geometry.prob_hi=0.064 0.032",
    f"amr.n_cell={nr if args.uniform else nr // 2} {nz}",
    f"amr.max_level={int(not args.uniform)}",
    "amr.max_grid_size=128",
    "amr.blocking_factor=4",
    "amr.n_error_buf=0",
    "boundary.field_lo=none periodic",
    "boundary.field_hi=pml periodic",
    "boundary.particle_lo=none periodic",
    "boundary.particle_hi=absorbing periodic",
    "boundary.verboncoeur_axis_correction=0",
    "warpx.n_rz_azimuthal_modes=1",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    "algo.current_deposition=esirkepov",
    "algo.particle_shape=3",
    "warpx.use_filter=1",
    "warpx.filter_npass_each_dir=0 1",
    "warpx.do_initial_div_cleaning=0",
    "warpx.n_field_gather_buffer=0",
    "warpx.n_current_deposition_buffer=0",
    f"warpx.const_dt={dt}",
    "warpx.verbose=0",
    "max_step=1",
    "warpx.current_injection=1",
    "warpx.current_injection.n_pairs=3",
    f"warpx.current_injection.file={profile}",
]
if not args.uniform:
    settings += [
        "amr.ref_ratio_vect=2 1",
        "warpx.fine_tag_lo=0 0",
        "warpx.fine_tag_hi=0.032 0.032",
        "warpx.conforming_radial_interface=1",
        f"warpx.conforming_radial_fixed_shape={int(not args.guarded_seed)}",
    ]
if args.restart:
    settings.append(f"amr.restart={args.restart}")
if args.guarded_seed:
    settings += [
        "diagnostics.diags_names=chk",
        "chk.diag_type=Full",
        "chk.format=checkpoint",
        "chk.intervals=0:1:1",
        "chk.file_prefix=diags/chk",
    ]
if args.plasma:
    settings += [
        "particles.species_names=ions",
        "ions.species_type=proton",
        "ions.injection_style=singleparticle",
        f"ions.single_particle_pos={0.05 if args.parent_particle else 0.015} 0 0.022",
        "ions.single_particle_u=0 0 0.02",
        "ions.single_particle_weight=1e9",
    ]
ri, ro = (0.044, 0.052) if args.parent_coil else (0.028, 0.036)
length_z = 0.008
for k in range(3):
    p = f"warpx.current_injection.pair_{k}.drive."
    settings += [
        f"{p}xlo={ri}",
        f"{p}xhi={ro}",
        f"{p}ylo=-0.5",
        f"{p}yhi=0.5",
        f"{p}zlo=0.008",
        f"{p}zhi=0.016",
        f"{p}dir={k}",
        f"{p}sign={-1 if k == 2 else 1}",
    ]
    if k:
        area = (ro - ri) * length_z if k == 1 else np.pi * (ro**2 - ri**2)
        settings.append(f"{p}A={area:.17g}")

libwarpx.initialize(settings)
libwarpx.warpx.evolve(1)
if args.guarded_seed:
    for step in (0, 1):
        chk = Path(f"diags/chk{step:06d}")
        assert (chk / "WarpXHeader").is_file()
        assert not (chk / "RadialFixedShape").exists()
    libwarpx.finalize()
    raise SystemExit(0)
fields = libwarpx.warpx.multifab_register()
seam = 32
nodes = h * (
    np.arange(nr + 1)
    if args.uniform
    else np.r_[np.arange(seam + 1), np.arange(seam + 2, nr + 1, 2)]
)
mid = (nodes[:-1] + nodes[1:]) / 2
dual_bounds = np.r_[0.0, mid, radius]
weights = (
    2 * np.pi * mid * np.diff(nodes) * dz,
    2 * np.pi * nodes * np.diff(dual_bounds) * dz,
    np.pi * np.diff(dual_bounds**2) * dz,
)
proton = (
    1.602176634e-19 * 1e9 * 299792458.0 * 0.02 / np.sqrt(1 + 0.02**2)
    if args.plasma
    else 0
)
for k, (direction, target) in enumerate(
    zip(
        ("r", "theta", "z"),
        (ro - ri, np.pi * (ri + ro), -length_z + proton),
        strict=True,
    )
):
    for family, value in (
        ("current_fp", target),
        ("Efield_fp", -dt * target / 8.8541878188e-12),
    ):

        def grid(lev):
            return np.squeeze(
                np.asarray(fields.get(family, dir=direction, level=lev)[...])
            )[:, :nz]

        leaf = grid(0)
        if not args.uniform:
            inside = grid(1)
            leaf = np.concatenate((inside, leaf[seam // 2 + int(k != 0) :]), axis=0)
        result = np.sum(weights[k][:, None] * leaf)
        rtol = max(3e-9, 8 * np.finfo(leaf.dtype).eps)
        np.testing.assert_allclose(result, value, rtol=rtol, atol=1e-14)
libwarpx.finalize()
