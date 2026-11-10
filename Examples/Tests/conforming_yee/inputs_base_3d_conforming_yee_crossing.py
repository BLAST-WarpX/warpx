"""Measure weak Gauss balance from moving particles and native leaf electric fields."""

import argparse

import numpy as np
from native_yee_mesh import host, mesh_fields

from pywarpx import geometry, libwarpx
from pywarpx.particle_containers import ParticleContainerWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--inward", action="store_true")
args = parser.parse_args()
n = 32
length = 0.1
ratio = np.array([2, 2, 1])
dt = 0.4 * length / (2 * n * 299792458.0)
steps = 100
shape = 3 if args.inward else 1
geometry.dims = 3
settings = [
    "crossing-conforming-yee",
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
    "warpx.ref_patch_function(x,y,z)=if((x*x+y*y<0.021^2)*(abs(z)<0.019),1,0)",
    "warpx.composite_yee_surface=1",
    "warpx.composite_yee_fixed_shape_current=1",
    "warpx.composite_yee_fixed_shape_gather=1",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    f"algo.current_deposition={'villasenor' if args.inward else 'esirkepov'}",
    f"algo.particle_shape={shape}",
    f"warpx.use_filter={int(args.inward)}",
    "warpx.filter_npass_each_dir=1 1 1",
    "warpx.do_initial_div_cleaning=0",
    "warpx.do_dive_cleaning=0",
    "warpx.do_divb_cleaning=0",
    "boundary.field_lo=periodic periodic periodic",
    "boundary.field_hi=periodic periodic periodic",
    "particles.species_names=electrons protons",
    f"warpx.const_dt={dt:.17g}",
    f"max_step={steps}",
    "warpx.verbose=0",
    "amrex.the_arena_init_size=0",
]
centre = 0.032 if args.inward else 0.012
sign = -1 if args.inward else 1
for name, species, speed in (
    ("electrons", "electron", 0.32),
    ("protons", "proton", 0.27),
):
    settings.extend(
        [
            f"{name}.species_type={species}",
            f"{name}.injection_style=Nuniformpercell",
            f"{name}.num_particles_per_cell_each_dim=4 4 2",
            f"{name}.profile=parse_density_function",
            f"{name}.density_function(x,y,z)=1e13*((x-{centre})^2+y^2+z^2<0.003^2)",
            f"{name}.momentum_distribution_type=constant",
            f"{name}.ux={sign * speed / np.sqrt(1 - speed**2):.17g}",
        ]
    )
libwarpx.initialize(settings)
fields = libwarpx.warpx.multifab_register()
blocks, boxes, covered = mesh_fields(fields, n, ratio)
species = [ParticleContainerWrapper(name) for name in ("electrons", "protons")]
# A coarse piecewise-linear test function is representable on both grids.
coarse_nodes = np.linspace(0, length, n + 1)
coarse_phi = length / (2 * np.pi) * np.sin(2 * np.pi * coarse_nodes / length)
fine_phi = np.interp(np.linspace(0, length, 2 * n + 1), coarse_nodes, coarse_phi)
charge_phi = (
    np.convolve(fine_phi, [0.25, 0.5, 0.25], mode="same") if args.inward else fine_phi
)


def sample():
    field_term = 0.0
    charge_term = 0.0
    count = np.zeros((2, 2), dtype=int)
    for level, phi in enumerate((coarse_phi, fine_phi)):
        h = length / (n * (ratio if level else np.ones(3)))
        for i, (lo, hi) in enumerate(boxes[level]):
            ex = host(blocks[level, "E", 0][i][2])
            cell = 0.25 * (
                ex[:, :-1, :-1] + ex[:, 1:, :-1] + ex[:, :-1, 1:] + ex[:, 1:, 1:]
            )
            weighted = cell * (np.diff(phi)[lo[0] : hi[0] + 1] / h[0])[:, None, None]
            if not level:
                region = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
                weighted = weighted[~covered[region]]
            field_term += 8.8541878188e-12 * np.prod(h) * np.sum(weighted)
        for s, pc in enumerate(species):
            for x, weight in zip(
                pc.get_particle_x(level=level), pc.get_particle_weight(level=level)
            ):
                x, weight = host(x), host(weight)
                count[s, level] += len(x)
                position = (x + length / 2) * (2 * n / length)
                j = np.floor(position).astype(int)
                f = position - j
                if shape == 1:
                    value = (1 - f) * charge_phi[j] + f * charge_phi[j + 1]
                else:
                    value = (
                        (1 - f) ** 3 * charge_phi[j - 1]
                        + (4 - 6 * f**2 + 3 * f**3) * charge_phi[j]
                        + (1 + 3 * f + 3 * f**2 - 3 * f**3) * charge_phi[j + 1]
                        + f**3 * charge_phi[j + 2]
                    ) / 6
                charge_term += (2 * s - 1) * 1.602176634e-19 * np.dot(weight, value)
    return charge_term, field_term, count


q0, e0, original = sample()
assert original.min() == 0 and np.all(original.sum(axis=1) > 0)
for _ in range(4):
    libwarpx.warpx.evolve(steps // 4)
    charge, electric, counts = sample()
    np.testing.assert_array_equal(counts.sum(axis=1), original.sum(axis=1))
    assert abs(charge - q0) > 1e-19
    np.testing.assert_allclose(electric - e0, -(charge - q0), rtol=2e-10, atol=0)
destination = 1 if args.inward else 0
assert np.all(counts[:, destination] > original[:, destination])
libwarpx.finalize()
