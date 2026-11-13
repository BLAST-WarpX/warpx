"""Track filtered Villasenor charge through a parent-grid copper wall or an open face."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import callbacks, geometry, libwarpx
from pywarpx.particle_containers import (
    ParticleBoundaryBufferWrapper,
    ParticleContainerWrapper,
)

parser = argparse.ArgumentParser()
parser.add_argument("--uniform", action="store_true")
parser.add_argument("--open-face", choices=("lo", "hi"))
parser.add_argument("--esirkepov", action="store_true")
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
geometry.dims = 3
nx = 64 if args.uniform else 32
rz = 2 if args.open_face else 1
parent_nz = 32
fine_nz = parent_nz * rz
nz = fine_nz if args.uniform else parent_nz
dz = 0.1 / fine_nz
orientation = -1 if args.open_face == "lo" else 1
dt = 0.4 * (0.1 / 64) / 299792458.0
wall = 0.032
steps = 150 if args.open_face else 95
settings = [
    "parent-wall",
    "geometry.dims=3",
    "geometry.prob_lo=-0.05 -0.05 -0.05",
    "geometry.prob_hi=0.05 0.05 0.05",
    f"amr.n_cell={nx} {nx} {nz}",
    f"amr.max_level={int(not args.uniform)}",
    f"amr.ref_ratio_vect=2 2 {rz}",
    "amr.max_grid_size=128",
    "amr.blocking_factor=2",
    "amr.n_error_buf=0",
    "warpx.fine_tag_lo=-0.01 -0.01 -0.01",
    "warpx.fine_tag_hi=0.01 0.01 0.01",
    f"warpx.composite_yee_surface={int(not args.uniform)}",
    f"warpx.composite_yee_fixed_shape_current={int(not args.uniform)}",
    f"warpx.composite_yee_fixed_shape_gather={int(not args.uniform)}",
    "algo.maxwell_solver=yee",
    f"algo.current_deposition={'esirkepov' if args.esirkepov else 'villasenor'}",
    "algo.field_gathering=energy-conserving",
    "algo.particle_shape=3",
    "warpx.use_filter=1",
    "warpx.filter_npass_each_dir=1 1 1",
    "warpx.do_initial_div_cleaning=0",
    "warpx.do_dive_cleaning=0",
    "warpx.do_divb_cleaning=0",
    f"boundary.field_lo=periodic periodic {'pml' if args.open_face else 'pec'}",
    f"boundary.field_hi=periodic periodic {'pml' if args.open_face else 'pec'}",
    "boundary.particle_lo=periodic periodic absorbing",
    "boundary.particle_hi=periodic periodic absorbing",
    "particles.species_names=ions",
    "ions.species_type=proton",
    "ions.injection_style=singleparticle",
    f"ions.single_particle_pos=0.0271 0.0012 {orientation * 0.0025}",
    f"ions.single_particle_u=0.01 -0.01 {orientation * 0.7}",
    "ions.single_particle_weight=1",
    f"warpx.const_dt={dt:.17g}",
    "warpx.verbose=0",
    f"max_step={steps}",
]
if args.open_face:
    settings += [
        f"warpx.pml_ncell={8 if args.uniform else 4}",
        f"warpx.pml_delta={8 if args.uniform else 4}",
        f"ions.save_particles_at_z{args.open_face}=1",
    ]
else:
    settings += [
        "algo.em_solver_medium=macroscopic",
        "algo.macroscopic_sigma_method=backwardeuler",
        f"macroscopic.sigma_function(x,y,z)=6e7*(z>{wall})*(z<0.043)",
        "macroscopic.epsilon_function(x,y,z)=epsilon0",
        "macroscopic.mu_function(x,y,z)=mu0",
        f"warpx.eb_implicit_function=min(z-{wall},0.043-z)",
        "ions.save_particles_at_eb=1",
    ]
libwarpx.initialize(settings)
ions = ParticleContainerWrapper("ions")
boundary = ParticleBoundaryBufferWrapper()
fields = libwarpx.warpx.multifab_register()
charge = 1.602176634e-19


def z_positions():
    parts = ions.get_particle_z(level=0, copy_to_host=True)
    return np.concatenate(parts) if parts else np.empty(0)


def charge_above(z):
    # Keep the entire free-space particle shape, including its off-domain tail.
    pad = 6
    x = np.abs(np.arange(-pad, fine_nz + 1 + pad) - (z + 0.05) / dz)
    # Cubic B-spline, then one pass of the native z binomial current filter.
    shape = np.where(
        x < 1, (4 - 6 * x * x + 3 * x**3) / 6, np.maximum(2 - x, 0) ** 3 / 6
    )
    shape = (np.roll(shape, 1) + 2 * shape + np.roll(shape, -1)) / 4
    above = np.cumsum(shape[::-1])[::-1][pad + 1 : pad + 1 + fine_nz]
    return above.reshape(parent_nz, rz).mean(axis=1)


def check_open_gather():
    # Just before the push, reconstruct two virtual Yee components in the
    # external guard cells from native parent fields exchanged with its PML.
    for component, nodal_z in (("x", True), ("z", False)):

        def guard_line(name, lev, x, y):
            mf = fields.get(name, dir=component, level=lev)
            for mfi in mf:  # one box per level in this control
                ilo, jlo, klo = mfi.fabbox().small_end
                line = mf.array(mfi).to_xp(copy=False, order="F")[
                    x - ilo, y - jlo, :, 0
                ]
                return np.asarray(line.get() if hasattr(line, "get") else line), klo
            raise AssertionError("Missing native gather field")

        parent, p0 = guard_line("Efield_fp", 0, 24, 16)
        fine, f0 = guard_line("composite_fixed_E", 1, 48, 32)
        nodes = (
            np.array([-3, -2, -1])
            if args.open_face == "lo"
            else fine_nz + np.array([1, 2, 3])
        )
        base, rem = np.divmod(nodes, rz)
        reference = parent[base - p0]
        if nodal_z:
            reference = reference + (rem / rz) * (parent[base + 1 - p0] - reference)
        peak = np.max(np.abs(reference))
        assert peak > 0.0
        np.testing.assert_allclose(
            fine[nodes - f0], reference, atol=2e-13 * peak, rtol=0
        )


previous_z = z_positions().item()
previous = charge_above(previous_z)
integrated = np.zeros(parent_nz)
pushed = []
hit = None
peak_continuity = 0.0
for step in range(1, steps + 1):
    probe = args.open_face and not args.uniform and step == 120
    if probe:
        callbacks.installcallback("particleinjection", check_open_gather)
    libwarpx.warpx.evolve(1)
    if probe:
        callbacks.uninstallcallback("particleinjection", check_open_gather)
    jz = np.squeeze(np.asarray(fields.get("current_fp", dir="z", level=0)[...]))
    # x and y are periodic; omit the duplicate nodes at their upper boundaries.
    transported = dt * (0.1 / nx) ** 2 / charge * jz[:nx, :nx, :].sum(axis=(0, 1))
    if args.uniform:
        transported = transported.reshape(parent_nz, rz).mean(axis=1)
    integrated += transported
    pushed.append(transported)
    z = z_positions()
    if z.size:
        assert z.size == 1 and hit is None
        previous_z = z.item()
        q = charge_above(previous_z)
        peak_continuity = max(peak_continuity, np.max(abs(q - previous - transported)))
        previous = q
    elif hit is None:
        hit = step
        if args.open_face:
            # Recover the endpoint of the last deposited trajectory using the
            # removed particle's momentum, including its off-domain charge tail.
            u = np.array(
                [
                    np.concatenate(
                        boundary.get_particle_boundary_buffer(
                            "ions", f"z_{args.open_face}", comp, 0
                        )
                    ).item()
                    for comp in ("ux", "uy", "uz")
                ]
            )
            last_z = previous_z + dt * u[2] / np.sqrt(1 + (u @ u) / 299792458.0**2)
            assert orientation * last_z >= 0.05
            residual = charge_above(last_z) - previous - transported
            peak_continuity = max(peak_continuity, np.max(abs(residual)))
assert hit is not None
assert (
    boundary.get_particle_boundary_buffer_size(
        "ions", f"z_{args.open_face}" if args.open_face else "eb"
    )
    == 1
)
# The current passed a complete macrocharge through a vacuum plane before collection.
measuring_plane = 10 if orientation < 0 else 21
np.testing.assert_allclose(integrated[measuring_plane], orientation, atol=2e-11, rtol=0)
assert peak_continuity < 2e-11
pushed = np.asarray(pushed)
if args.reference:
    with np.load(args.reference) as ref:
        assert int(ref["hit"]) == hit
        np.testing.assert_allclose(pushed, ref["pushed"], atol=2e-11, rtol=0)
else:
    np.savez("wall_reference.npz", hit=hit, pushed=pushed)
print(
    "Absorbed at step",
    hit,
    "upstream collected charge fraction",
    integrated[measuring_plane],
    "maximum plane charge residual",
    peak_continuity,
)
libwarpx.finalize()
