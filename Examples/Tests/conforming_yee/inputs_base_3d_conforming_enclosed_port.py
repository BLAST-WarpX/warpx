"""Compare a driven, conducting split ring on the finest Yee grid with a uniform mesh."""

import argparse
from pathlib import Path

import numpy as np
from native_yee_mesh import host, native_blocks

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--uniform", action="store_true")
parser.add_argument("--expand-aperture", action="store_true")
parser.add_argument("--impressed-current", action="store_true")
parser.add_argument("--reference", type=Path)
parser.add_argument("--touch", choices=("material", "eb", "port"))
args = parser.parse_args()
geometry.dims = 3
nx = 64 if args.uniform else 32
h = 0.1 / 64
dt = 0.3 * h / 299792458.0
Path("coil_waveform.txt").write_text(f"0 0\n{6 * dt:.17g} 12\n{100 * dt:.17g} 12\n")
r = "sqrt(x*x+y*y)"
ring = (
    f"min(min(min({r}-0.009375,0.015625-{r}),0.003125-abs(z)),max(abs(x)-0.003125,y))"
)
near_join = "min(0.002-abs(x-0.03125),min(0.004-abs(y),0.004-abs(z)))"
eb = f"max({ring},{near_join})" if args.touch == "eb" else ring
material = f"max({ring},{near_join})" if args.touch == "material" else ring
# The normal contour encloses the copper footprint in the gap at negative y.
port_lo, port_hi = (-0.035, -0.0225) if args.touch == "port" else (-0.01875, -0.00625)
# A subcell expansion leaves the same Yee nodes in an inclusive aperture.
pad = h * 1.0e-6 if args.expand_aperture else 0.0
port_lo, port_hi = port_lo - pad, port_hi + pad
z_bound = 0.00625 + pad
settings = [
    "enclosed-port",
    "geometry.dims=3",
    "geometry.prob_lo=-0.05 -0.05 -0.05",
    "geometry.prob_hi=0.05 0.05 0.05",
    f"amr.n_cell={nx} {nx} 64",
    f"amr.max_level={int(not args.uniform)}",
    "amr.ref_ratio_vect=2 2 1",
    "amr.max_grid_size=16",
    "amr.blocking_factor=4",
    "amr.n_error_buf=0",
    "warpx.fine_tag_lo=-0.03125 -0.03125 -0.03125",
    "warpx.fine_tag_hi=0.03125 0.03125 0.03125",
    f"warpx.composite_yee_surface={int(not args.uniform)}",
    "algo.maxwell_solver=yee",
    "algo.em_solver_medium=macroscopic",
    "algo.macroscopic_sigma_method=backwardeuler",
    f"warpx.eb_implicit_function={eb}",
    f"macroscopic.sigma_function(x,y,z)=6e7*({material}>0)",
    "macroscopic.epsilon_function(x,y,z)=epsilon0",
    "macroscopic.mu_function(x,y,z)=mu0",
    "warpx.do_initial_div_cleaning=0",
    "warpx.do_dive_cleaning=0",
    "warpx.do_divb_cleaning=0",
    "boundary.field_lo=periodic periodic periodic",
    "boundary.field_hi=periodic periodic periodic",
    "warpx.current_controlled_port=1",
    "warpx.current_controlled_port.file=coil_waveform.txt",
    "warpx.current_controlled_port.direction=0",
    f"warpx.current_controlled_port.terminal_0.lower_bound=-0.003125 {port_lo} {-z_bound}",
    f"warpx.current_controlled_port.terminal_0.upper_bound=-0.003125 {port_hi} {z_bound}",
    f"warpx.current_controlled_port.terminal_1.lower_bound=0.003125 {port_lo} {-z_bound}",
    f"warpx.current_controlled_port.terminal_1.upper_bound=0.003125 {port_hi} {z_bound}",
    f"warpx.const_dt={dt:.17g}",
    "max_step=6",
    "warpx.verbose=0",
]
if args.impressed_current:
    settings.extend(
        [
            f"warpx.composite_yee_fixed_shape_current={int(not args.uniform)}",
            f"warpx.composite_yee_fixed_shape_gather={int(not args.uniform)}",
            "algo.current_deposition=villasenor",
            "algo.field_gathering=energy-conserving",
            "algo.particle_shape=3",
            "warpx.use_filter=0",
            "warpx.n_current_deposition_buffer=0",
            "warpx.n_field_gather_buffer=0",
            "particles.species_names=ions",
            "ions.species_type=proton",
            "ions.injection_style=singleparticle",
            "ions.single_particle_pos=0.009 0.006 0.0064",
            "ions.single_particle_u=0 0.02 0",
            "ions.single_particle_weight=1.0e9",
            "warpx.current_injection=1",
            "warpx.current_injection.file=coil_waveform.txt",
            "warpx.current_injection.pair_0.drive.xlo=-0.003125",
            "warpx.current_injection.pair_0.drive.xhi=0.003125",
            "warpx.current_injection.pair_0.drive.ylo=-0.015625",
            "warpx.current_injection.pair_0.drive.yhi=-0.009375",
            "warpx.current_injection.pair_0.drive.zlo=-0.003125",
            "warpx.current_injection.pair_0.drive.zhi=0.003125",
            "warpx.current_injection.pair_0.drive.A=0.00625*0.00625",
            "warpx.current_injection.pair_0.drive.dir=0",
        ]
    )
libwarpx.initialize(settings)
fields = libwarpx.warpx.multifab_register()
level = int(not args.uniform)
start = np.array([20, 18, 24])
end = np.array([45, 45, 41])


def snapshot(family, component):
    # Collect valid native values even where the ring crosses a fine FAB seam.
    out = np.full(end - start, np.nan)
    for lo, hi, view in native_blocks(fields, level, family, component):
        low, high = np.maximum(lo, start), np.minimum(hi + 1, end)
        if np.all(high > low):
            src = tuple(slice(i, j) for i, j in zip(low - lo, high - lo))
            dest = tuple(slice(i, j) for i, j in zip(low - start, high - start))
            out[dest] = host(view[src])
    assert np.isfinite(out).all()
    return out


saved = {}
for step in range(1, 7):
    libwarpx.warpx.evolve(1)
    if step not in (1, 3, 6):
        continue
    for family in "EB":
        for component, direction in enumerate("xyz"):
            saved[f"{family}{direction}_{step}"] = snapshot(family, component)
    if args.impressed_current:
        jx, jy = (snapshot("j", d) for d in (0, 1))
        saved[f"jx_{step}"], saved[f"jy_{step}"] = jx, jy
        if step == 1:
            # Initial fields vanish. The antenna supplies a 1 A step-average
            # along x, while the independent proton carries current along y.
            np.testing.assert_allclose(jx.sum() * h**3 / 0.00625, 1.0, rtol=2e-9)
            proton_current_length = (
                1.602176634e-19 * 1.0e9 * 299792458.0 * 0.02 / np.sqrt(1 + 0.02**2)
            )
            np.testing.assert_allclose(
                jy.sum() * h**3, proton_current_length, rtol=2e-9
            )
    by, bz = saved[f"By_{step}"], saved[f"Bz_{step}"]
    i, j0, j1, k0, k1 = (
        32 - start[0],
        20 - start[1],
        28 - start[1],
        28 - start[2],
        36 - start[2],
    )
    loop = (by[i, j0 : j1 + 1, k0 - 1] - by[i, j0 : j1 + 1, k1]).sum() + (
        bz[i, j1, k0 : k1 + 1] - bz[i, j0 - 1, k0 : k1 + 1]
    ).sum()
    mu0 = 1.2566370612685e-6
    np.testing.assert_allclose(h * loop / mu0, 2.0 * step, rtol=2e-9)
assert np.max(np.abs(saved["Ex_6"])) > 0
if args.reference:
    with np.load(args.reference) as ref:
        for key, field in saved.items():
            peak = max(np.max(np.abs(ref[key])), np.finfo(float).tiny)
            np.testing.assert_allclose(
                field, ref[key], atol=2e-11 * peak, rtol=0, err_msg=key
            )
else:
    np.savez("enclosed_reference.npz", **saved)
libwarpx.finalize()
