"""Compare the active Yee leaves to a uniform grid after a pulse crosses both slab faces."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx

p = argparse.ArgumentParser()
p.add_argument("--ratio", nargs=2, type=int, required=True)
p.add_argument("--polarization", choices=("te", "tm"), required=True)
p.add_argument("--oblique", action="store_true")
p.add_argument("--reference", type=Path)
p.add_argument("--checkpoint", action="store_true")
p.add_argument("--restart", type=Path)
p.add_argument("--split-reference", type=Path)
p.add_argument("--ordinary", action="store_true")
p.add_argument("--particle-at-interface", action="store_true")
args = p.parse_args()
rx, rz = args.ratio
joined = (rx, rz) != (1, 1)
geometry.dims = 2
nx, nz, h = 240, 48, 0.0005
steps = 150
dt = 0.48 * h / 299792458.0
settings = [
    "conforming-xz-pulse",
    "geometry.dims=2",
    "geometry.prob_lo=-0.06 -0.012",
    "geometry.prob_hi=0.06 0.012",
    f"amr.n_cell={nx // rx} {nz // rz}",
    f"amr.max_level={int(joined)}",
    f"amr.ref_ratio_vect={rx} {rz}",
    "amr.max_grid_size=512",
    "amr.blocking_factor=2",
    "amr.n_error_buf=0",
    "warpx.fine_tag_lo=-0.02 -0.012",
    "warpx.fine_tag_hi=0.02 0.012",
    f"warpx.composite_xz_interface={int(joined and not args.ordinary)}",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    "warpx.n_field_gather_buffer=0",
    "warpx.n_current_deposition_buffer=0",
    "warpx.use_filter=0",
    "warpx.do_initial_div_cleaning=0",
    "boundary.field_lo=pml periodic",
    "boundary.field_hi=pml periodic",
    "warpx.pml_ncell=8",
    "warpx.pml_delta=8",
    f"warpx.const_dt={dt:.17g}",
    f"max_step={steps}",
    "warpx.verbose=0",
]
if args.checkpoint:
    settings.extend(
        [
            "diagnostics.diags_names=chk",
            "chk.diag_type=Full",
            "chk.format=checkpoint",
            f"chk.intervals={steps // 2}",
            "chk.file_prefix=diags/chk",
        ]
    )
if args.restart:
    settings.append(f"amr.restart={args.restart}")
if args.particle_at_interface:
    settings.extend(
        [
            "particles.species_names=ions",
            "algo.particle_shape=2",
            "algo.current_deposition=esirkepov",
            "ions.species_type=proton",
            "ions.injection_style=singleparticle",
            "ions.single_particle_pos=0.02 0 0",
            "ions.single_particle_u=0.08 0 0",
            "ions.single_particle_weight=1e6",
        ]
    )
libwarpx.initialize(settings)
register = libwarpx.warpx.multifab_register()
components = ("Ey", "Bx", "Bz") if args.polarization == "te" else ("By", "Ex", "Ez")
levels = (0, 1) if joined else (0,)


stagger = {
    "Ey": (1, 1),
    "Bx": (1, 0),
    "Bz": (0, 1),
    "By": (0, 0),
    "Ex": (0, 1),
    "Ez": (1, 0),
}


def field(comp, level):
    mf = register.get(comp[0] + "field_fp", dir=comp[1].lower(), level=level)
    a = np.asarray(mf[...]).copy()
    v = np.squeeze(a)
    start = np.array(mf.box_array().minimal_box().small_end)
    nodal = stagger[comp]
    dx, dz = (h, h) if level else (rx * h, rz * h)
    x = -0.06 + (start[0] + np.arange(v.shape[0]) + 0.5 * (1 - nodal[0])) * dx
    z = -0.012 + (start[1] + np.arange(v.shape[1]) + 0.5 * (1 - nodal[1])) * dz
    return mf, a, x, z


# Both polarizations start divergence-free. The compact source is completely
# inside the fine region. Its two wave fronts must reach the exterior.
if not args.restart:
    for lev in levels:
        mf, data, x, z = field(components[0], lev)
        envelope = np.where(abs(x) < 0.012, np.cos(np.pi * x / 0.024) ** 6, 0.0)
        mode = np.cos(2 * np.pi * z / 0.024) if args.oblique else np.ones_like(z)
        np.squeeze(data)[:] = envelope[:, None] * mode[None, :]
        mf[...] = data
libwarpx.warpx.evolve(steps - libwarpx.warpx.getistep(0))
snap = {}
for comp in stagger:
    for level in levels:
        _, a, x, z = field(comp, level)
        snap[comp, level] = (np.squeeze(a), x, z)

if args.checkpoint:
    assert Path("diags/chk000075/CompositeXZSlab").is_file() == (
        joined and not args.ordinary
    )
    np.savez(
        "slab_endpoint.npz",
        **{f"{comp}_{lev}": values[0] for (comp, lev), values in snap.items()},
    )
if args.split_reference:
    with np.load(args.split_reference) as ref:
        for (comp, lev), (v, _, _) in snap.items():
            expected = ref[f"{comp}_{lev}"]
            peak = np.max(abs(expected))
            # Native restart preserves the dynamical fields on both levels.
            np.testing.assert_allclose(v, expected, rtol=0, atol=2e-12 * peak)

if not joined:
    data = {comp: snap[comp, 0][0] for comp in components}
    np.savez("pulse.npz", **data)
elif not args.ordinary:
    assert args.reference
    c = 299792458.0
    errors = []
    exterior_energy = total_energy = 0.0
    with np.load(args.reference) as ref:
        for comp in components:
            # Match the reference to each native Yee point, including midpoints
            # between two fine face centers. z is periodic.
            expected = ref[comp]
            n0 = stagger[comp]
            xu = -0.06 + (np.arange(expected.shape[0]) + 0.5 * (1 - n0[0])) * h
            zu = -0.012 + (np.arange(nz) + 0.5 * (1 - n0[1])) * h
            _, xf, _ = snap[comp, 1]
            ref_norm = err_norm = 0.0
            for lev in levels:
                v, x, z = snap[comp, lev]
                rows = (
                    (x < xf[0] - 1e-13) | (x > xf[-1] + 1e-13)
                    if lev == 0
                    else np.ones(len(x), bool)
                )
                z = z[: nz // (rz if lev == 0 else 1)]
                v = v[rows, : len(z)]
                pick_z = np.stack(
                    [np.interp(z, zu, row[:nz], period=0.024) for row in expected]
                )
                sample = np.stack([np.interp(x[rows], xu, col) for col in pick_z.T]).T
                area = h * h * (rx * rz if lev == 0 else 1)
                ref_norm += area * np.sum(sample**2)
                err_norm += area * np.sum((v - sample) ** 2)
                weight = c * c if comp.startswith("B") else 1.0
                total_energy += weight * area * np.sum(v**2)
                if lev == 0:
                    exterior_energy += weight * area * np.sum(v**2)
            if ref_norm > 0:
                errors.append(np.sqrt(err_norm / ref_norm))
            else:
                assert err_norm < 1e-25
    assert exterior_energy > 0.12 * total_energy
    print(
        "Relative field errors",
        errors,
        "exterior energy fraction",
        exterior_energy / total_energy,
    )
    if not args.oblique and rx == 1:
        assert max(errors) < 1e-10
    else:
        assert max(errors) < 0.04
libwarpx.finalize()
