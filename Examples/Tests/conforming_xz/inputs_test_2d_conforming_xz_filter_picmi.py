"""Check filtered charge continuity on each physical side of a Cartesian Yee slab."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx
from pywarpx.particle_containers import ParticleContainerWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--near-interface", action="store_true")
parser.add_argument("--restart", type=Path)
parser.add_argument("--change-filter", action="store_true")
parser.add_argument("--fixed-shape", action="store_true")
args = parser.parse_args()
geometry.dims = 2
dt = 0.4e-3 / 299792458.0
fine_x = 0.038 if args.near_interface else 0.0484
settings = [
    "filtered-slab",
    "geometry.dims=2",
    "geometry.prob_lo=0 0",
    "geometry.prob_hi=0.096 0.016",
    "amr.n_cell=48 8",
    "amr.max_level=1",
    "amr.ref_ratio_vect=2 2",
    "amr.max_grid_size=128",
    "amr.blocking_factor=2",
    "amr.n_error_buf=0",
    "warpx.fine_tag_lo=0.032 0",
    "warpx.fine_tag_hi=0.064 0.016",
    "warpx.composite_xz_interface=1",
    f"warpx.composite_xz_fixed_shape={int(args.fixed_shape)}",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    "algo.current_deposition=esirkepov",
    "algo.particle_shape=2",
    "warpx.n_field_gather_buffer=0",
    "warpx.n_current_deposition_buffer=0",
    "warpx.do_initial_div_cleaning=0",
    "warpx.use_filter=1",
    f"warpx.filter_npass_each_dir={1 if args.change_filter else 2} 1",
    "boundary.field_lo=periodic periodic",
    "boundary.field_hi=periodic periodic",
    "boundary.particle_lo=periodic periodic",
    "boundary.particle_hi=periodic periodic",
    "particles.species_names=ions",
    "ions.species_type=proton",
    "ions.injection_style=multipleparticles",
    f"ions.multiple_particles_pos_x=0.0126 {fine_x}",
    "ions.multiple_particles_pos_y=0 0",
    "ions.multiple_particles_pos_z=0.0159 0.0159",
    "ions.multiple_particles_ux=0.21 -0.22",
    "ions.multiple_particles_uy=0.1 0.15",
    "ions.multiple_particles_uz=0.36 0.39",
    "ions.multiple_particles_weight=1e6 1e6",
    f"warpx.const_dt={dt:.17g}",
    "warpx.verbose=0",
    "max_step=4",
]
if args.restart:
    settings.append(f"amr.restart={args.restart}")
else:
    settings.extend(
        [
            "diagnostics.diags_names=chk",
            "chk.diag_type=Full",
            "chk.format=checkpoint",
            "chk.intervals=2",
            "chk.file_prefix=diags/chk",
        ]
    )
libwarpx.initialize(settings)
register = libwarpx.warpx.multifab_register()
ions = ParticleContainerWrapper("ions")
charge = 1.602176634e-13


def tsc(distance):
    a = np.abs(distance)
    return np.where(a < 0.5, 0.75 - a**2, 0.5 * np.maximum(1.5 - a, 0.0) ** 2)


def binomial(values):
    return (np.roll(values, 1) + 2 * values + np.roll(values, -1)) / 4


def node_charge(lev):
    h = 0.002 / (1 + lev)
    nz = 8 * (1 + lev)
    nodes = np.arange(49) if lev == 0 else np.arange(32, 65)
    x = np.concatenate(ions.get_particle_x(level=lev, copy_to_host=True))
    z = np.concatenate(ions.get_particle_z(level=lev, copy_to_host=True))
    assert x.size == z.size == 1
    wx = binomial(binomial(tsc(nodes - x.item() / h)))
    wz = binomial(tsc((np.arange(nz) - z.item() / h + nz / 2) % nz - nz / 2))
    return charge * wx[:, None] * wz, z.item()


def current_flux(lev):
    h = 0.002 / (1 + lev)
    nz = 8 * (1 + lev)
    jx, jz = [
        np.squeeze(np.asarray(register.get("current_fp", dir=d, level=lev)[...]))[
            :, :nz
        ]
        for d in ("x", "z")
    ]
    return h * (
        np.diff(np.pad(jx, ((1, 1), (0, 0))), axis=0) + jz - np.roll(jz, 1, axis=1)
    )


previous = [node_charge(lev)[0] for lev in (0, 1)]
peak = 0.0
for _ in range(libwarpx.warpx.getistep(0), 4):
    libwarpx.warpx.evolve(1)
    for lev in (0, 1):
        q, z = node_charge(lev)
        flux = current_flux(lev)
        residual = q - previous[lev] + dt * flux
        # The parent stores a restricted copy of the source under the slab.
        active = np.r_[0:16, 33:49] if lev == 0 else slice(None)
        peak = max(peak, np.max(np.abs(residual[active])) / charge)
        assert np.max(np.abs(dt * flux[active])) > 1e-3 * charge
        assert np.sum(q) / charge > 0.9999
        previous[lev] = q
for lev in (0, 1):
    assert (
        node_charge(lev)[1] < 0.001
    )  # Both particles crossed the periodic z boundary.
print("Maximum filtered charge-balance residual / particle charge", peak)
assert peak < max(2e-11, 128 * np.finfo(flux.dtype).eps)
final = {
    f"{name}_{direction}_{lev}": np.asarray(
        register.get(name, dir=direction, level=lev)[...]
    )
    for lev in (0, 1)
    for name in ("Efield_fp", "Bfield_fp", "current_fp")
    for direction in ("x", "y", "z")
}
for lev in (0, 1):
    final[f"charge_{lev}"] = previous[lev]
if args.restart:
    with np.load(args.restart.parents[1] / "filtered_endpoint.npz") as ref:
        for key, value in final.items():
            np.testing.assert_allclose(
                value,
                ref[key],
                rtol=0,
                atol=8 * np.finfo(value.dtype).eps * np.max(abs(ref[key])),
            )
else:
    np.savez("filtered_endpoint.npz", **final)
libwarpx.finalize()
