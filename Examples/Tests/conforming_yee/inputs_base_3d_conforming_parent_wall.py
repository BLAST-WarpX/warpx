"""Compare a filtered Villasenor source at a parent-grid copper wall with a uniform Yee mesh."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import geometry, libwarpx
from pywarpx.particle_containers import (
    ParticleBoundaryBufferWrapper,
    ParticleContainerWrapper,
)

parser = argparse.ArgumentParser()
parser.add_argument("--uniform", action="store_true")
parser.add_argument("--esirkepov", action="store_true")
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
geometry.dims = 3
nx = 64 if args.uniform else 32
nz = 32
dz = 0.1 / nz
dt = 0.4 * (0.1 / 64) / 299792458.0
wall = 0.032
steps = 95
settings = [
    "parent-wall",
    "geometry.dims=3",
    "geometry.prob_lo=-0.05 -0.05 -0.05",
    "geometry.prob_hi=0.05 0.05 0.05",
    f"amr.n_cell={nx} {nx} {nz}",
    f"amr.max_level={int(not args.uniform)}",
    "amr.ref_ratio_vect=2 2 1",
    "amr.max_grid_size=128",
    "amr.blocking_factor=2",
    "amr.n_error_buf=0",
    "warpx.fine_tag_lo=-0.01 -0.01 -0.01",
    "warpx.fine_tag_hi=0.01 0.01 0.01",
    f"warpx.composite_yee_surface={int(not args.uniform)}",
    f"warpx.composite_yee_fixed_shape_current={int(not args.uniform)}",
    f"warpx.composite_yee_fixed_shape_gather={int(not args.uniform)}",
    "algo.maxwell_solver=yee",
    "algo.em_solver_medium=macroscopic",
    "algo.macroscopic_sigma_method=backwardeuler",
    f"macroscopic.sigma_function(x,y,z)=6e7*(z>{wall})*(z<0.043)",
    "macroscopic.epsilon_function(x,y,z)=epsilon0",
    "macroscopic.mu_function(x,y,z)=mu0",
    f"warpx.eb_implicit_function=min(z-{wall},0.043-z)",
    f"algo.current_deposition={'esirkepov' if args.esirkepov else 'villasenor'}",
    "algo.field_gathering=energy-conserving",
    "algo.particle_shape=3",
    "warpx.use_filter=1",
    "warpx.filter_npass_each_dir=1 1 1",
    "warpx.do_initial_div_cleaning=0",
    "warpx.do_dive_cleaning=0",
    "warpx.do_divb_cleaning=0",
    "boundary.field_lo=periodic periodic pec",
    "boundary.field_hi=periodic periodic pec",
    "boundary.particle_lo=periodic periodic absorbing",
    "boundary.particle_hi=periodic periodic absorbing",
    "particles.species_names=ions",
    "ions.species_type=proton",
    "ions.injection_style=singleparticle",
    "ions.single_particle_pos=0.0271 0.0012 0.0025",
    "ions.single_particle_u=0.01 -0.01 0.7",
    "ions.single_particle_weight=1",
    "ions.save_particles_at_eb=1",
    f"warpx.const_dt={dt:.17g}",
    "warpx.verbose=0",
    f"max_step={steps}",
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
    x = np.abs(np.arange(nz + 1) - (z + 0.05) / dz)
    # Cubic B-spline, then one pass of the native z binomial current filter.
    shape = np.where(
        x < 1, (4 - 6 * x * x + 3 * x**3) / 6, np.maximum(2 - x, 0) ** 3 / 6
    )
    shape = (np.roll(shape, 1) + 2 * shape + np.roll(shape, -1)) / 4
    return np.cumsum(shape[:0:-1])[::-1]


previous = charge_above(z_positions().item())
integrated = np.zeros(nz)
pushed = []
hit = None
peak_continuity = 0.0
for step in range(1, steps + 1):
    libwarpx.warpx.evolve(1)
    jz = np.squeeze(np.asarray(fields.get("current_fp", dir="z", level=0)[...]))
    # x and y are periodic; omit the duplicate nodes at their upper boundaries.
    transported = dt * (0.1 / nx) ** 2 / charge * jz[:nx, :nx, :].sum(axis=(0, 1))
    integrated += transported
    pushed.append(transported)
    z = z_positions()
    if z.size:
        assert z.size == 1 and hit is None
        q = charge_above(z.item())
        peak_continuity = max(peak_continuity, np.max(abs(q - previous - transported)))
        previous = q
    elif hit is None:
        hit = step
assert hit is not None
assert boundary.get_particle_boundary_buffer_size("ions", "eb") == 1
# The current passed a complete macrocharge through a vacuum plane before collection.
np.testing.assert_allclose(integrated[21], 1, atol=2e-11, rtol=0)
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
    integrated[21],
    "maximum pre-impact plane charge residual",
    peak_continuity,
)
libwarpx.finalize()
