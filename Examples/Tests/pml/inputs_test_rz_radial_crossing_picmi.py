"""Check cellwise cylindrical charge continuity while a kinetic particle crosses a Yee join."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx
from pywarpx.particle_containers import ParticleContainerWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--inward", action="store_true")
parser.add_argument("--low-seam", action="store_true")
parser.add_argument("--shape", type=int, choices=(1, 2, 3), default=3)
parser.add_argument("--filter", action="store_true")
parser.add_argument("--restart", type=Path)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
geometry.dims = "RZ"
n, nz, radius, length = 32, 24, 0.08, 0.048
h, dz = radius / (2 * n), length / nz
velocity = -0.38 if args.inward else 0.38
z_direction = -1 if args.low_seam else 1
dt = 0.45 * h / 299792458.0
steps = 64
settings = [
    "radial-fixed-shape-crossing",
    "geometry.dims=RZ",
    f"geometry.prob_lo=0 {-length / 2}",
    f"geometry.prob_hi={radius} {length / 2}",
    f"amr.n_cell={n} {nz}",
    "amr.max_level=1",
    "amr.ref_ratio_vect=2 1",
    "amr.blocking_factor=4",
    "amr.max_grid_size=128",
    "amr.n_error_buf=0",
    f"warpx.fine_tag_lo=0 {-length / 2}",
    f"warpx.fine_tag_hi={radius / 2} {length / 2}",
    "warpx.conforming_radial_interface=1",
    "warpx.conforming_radial_fixed_shape=1",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    "algo.current_deposition=esirkepov",
    f"algo.particle_shape={args.shape}",
    f"warpx.use_filter={int(args.filter)}",
    "warpx.filter_npass_each_dir=0 1",
    "warpx.n_rz_azimuthal_modes=1",
    "warpx.do_initial_div_cleaning=0",
    "boundary.field_lo=none periodic",
    "boundary.field_hi=pml periodic",
    "boundary.particle_lo=none periodic",
    "boundary.particle_hi=absorbing periodic",
    "particles.species_names=ions",
    "ions.species_type=proton",
    "ions.injection_style=singleparticle",
    f"ions.single_particle_pos={radius / 2 + (4 if args.inward else -4) * h} 0 {z_direction * length * 0.41}",
    f"ions.single_particle_u={velocity} 0.13 {z_direction * 0.28}",
    "ions.single_particle_weight=1e7",
    f"warpx.const_dt={dt:.17g}",
    "warpx.verbose=0",
    f"max_step={steps}",
]
if args.inward:
    settings += [
        "diagnostics.diags_names=chk",
        "chk.diag_type=Full",
        "chk.format=checkpoint",
        f"chk.intervals={steps // 2}",
        "chk.file_prefix=diags/chk",
    ]
if args.restart:
    settings.append(f"amr.restart={args.restart}")
libwarpx.initialize(settings)
fields = libwarpx.warpx.multifab_register()
pc = ParticleContainerWrapper("ions")


def values(name, direction, level):
    mf = fields.get(name, dir=direction, level=level)
    return np.squeeze(np.asarray(mf[...]))[:, :nz]


seam = values("Efield_fp", "r", 1).shape[0]
rnodes = h * np.r_[np.arange(seam + 1), np.arange(seam + 2, 2 * n + 1, 2)]
rface = (rnodes[1:] + rnodes[:-1]) / 2
area = np.pi * np.diff(np.r_[0.0, rface, radius] ** 2)
charge = 1.602176634e-19 * 1e7
eps0 = 8.8541878188e-12


def shape(distance):
    a = abs(distance)
    if args.shape == 1:
        return np.maximum(1 - a, 0)
    if args.shape == 2:
        return np.where(a < 0.5, 0.75 - a**2, 0.5 * np.maximum(1.5 - a, 0) ** 2)
    return np.where(a < 1, (4 - 6 * a**2 + 3 * a**3) / 6, np.maximum(2 - a, 0) ** 3 / 6)


def particle_charge():
    locations = []
    for lev in (0, 1):
        for rr, zz in zip(
            pc.get_particle_r(lev, copy_to_host=True),
            pc.get_particle_z(lev, copy_to_host=True),
            strict=True,
        ):
            locations.extend(
                (float(r), float(z), lev) for r, z in zip(rr, zz, strict=True)
            )
    assert len(locations) == 1
    r, z, level = locations[0]
    wr = shape(np.arange(2 * n + 1) - r / h)
    distance = np.arange(nz) - (z + length / 2) / dz
    wz = shape((distance + nz / 2) % nz - nz / 2)
    if args.filter:
        wz = (np.roll(wz, 1) + 2 * wz + np.roll(wz, -1)) / 4
    composite = np.r_[wr[: seam + 1], wr[seam + 2 :: 2]]
    mid = wr[seam + 1 :: 2]
    composite[seam:-1] += 0.5 * mid
    composite[seam + 1 :] += 0.5 * mid
    return charge * composite[:, None] * wz, level


def divergence(name):
    jr = np.vstack((values(name, "r", 1), values(name, "r", 0)[seam // 2 :]))
    jz = np.vstack((values(name, "z", 1), values(name, "z", 0)[seam // 2 + 1 :]))
    radial_flux = 2 * np.pi * dz * rface[:, None] * jr
    return np.diff(np.pad(radial_flux, ((1, 1), (0, 0))), axis=0) + area[:, None] * (
        jz - np.roll(jz, 1, axis=1)
    )


def work_error():
    # Independently integrate J.E on the virtual particle grid and on the active
    # Yee leaves. The source uses all three cylindrical current components.
    errors = []
    for direction in ("r", "theta", "z"):
        cell = direction == "r"
        radial = h * (np.arange(2 * n + int(not cell)) + 0.5 * cell)
        fine_volume = 2 * np.pi * radial * h * dz
        if cell:
            leaf_volume = 2 * np.pi * rface * np.diff(rnodes) * dz
        elif direction == "theta":
            widths = np.diff(np.r_[0.0, rface, radius])
            leaf_volume = 2 * np.pi * rnodes * widths * dz
        else:
            leaf_volume = area * dz
        virtual = (
            values("radial_fixed_J", direction, 1)
            * values("composite_fixed_E", direction, 1)
            * fine_volume[:, None]
        )
        physical = []
        for lev in (1, 0):
            product = values("current_fp", direction, lev) * values(
                "Efield_aux", direction, lev
            )
            physical.append(product if lev else product[seam // 2 + int(not cell) :])
        leaf = np.vstack(physical) * leaf_volume[:, None]
        scale = np.sum(abs(virtual)) + np.sum(abs(leaf))
        if scale:
            errors.append(abs(np.sum(virtual) - np.sum(leaf)) / scale)
    return max(errors, default=0.0)


q0, original = particle_charge()
e0 = divergence("Efield_fp")
last = q0
peak_continuity = peak_gauss = peak_work = 0.0
for step in range(libwarpx.warpx.getistep(0), steps):
    libwarpx.warpx.evolve(1)
    q, level = particle_charge()
    change = q - last
    peak_continuity = max(
        peak_continuity, np.max(abs(change + dt * divergence("current_fp"))) / charge
    )
    # The outer boundary node shares a dual cell with the split-field PML.
    # This field view contains the regular domain and ends before that radial flux.
    gauss = q - q0 - eps0 * (divergence("Efield_fp") - e0)
    peak_gauss = max(peak_gauss, np.max(abs(gauss[:-1])) / charge)
    if step % 4 == 3:
        peak_work = max(peak_work, work_error())
    last = q
print(
    f"Level {original} to {level}, peak continuity {peak_continuity:.3g}, Gauss {peak_gauss:.3g}, work {peak_work:.3g}"
)
if not args.restart:
    assert original == int(not args.inward)
assert level == int(args.inward)
assert peak_continuity < 2.0e-11
assert peak_gauss < 2.0e-10
assert peak_work < 2.0e-12
if args.inward:
    final = {
        f"{field}_{d}_{lev}": values(field, d, lev)
        for field in ("Efield_fp", "Bfield_fp")
        for d in ("r", "theta", "z")
        for lev in (0, 1)
    }
    final["charge"] = last
    if args.reference:
        with np.load(args.reference) as reference:
            for key, value in final.items():
                np.testing.assert_allclose(
                    value,
                    reference[key],
                    atol=2.0e-12 * np.max(abs(reference[key])),
                    rtol=2.0e-12,
                )
    else:
        np.savez("radial_crossing.npz", **final)
libwarpx.finalize()
