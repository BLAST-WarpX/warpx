"""Check the electric-field callback after every level in an explicit step."""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--medium", choices=["vacuum", "backwardeuler", "laxwendroff"])
parser.add_argument("--levels", type=int, choices=[1, 2], default=1)
args = parser.parse_args()
medium = "vacuum" if args.medium == "vacuum" else "macroscopic"
parameters = [
    "geometry.dims=3",
    "geometry.prob_lo=0 0 0",
    "geometry.prob_hi=.032 .032 .032",
    "amr.n_cell=16 16 16",
    "amr.max_grid_size=32",
    "amr.blocking_factor=8",
    f"amr.max_level={args.levels - 1}",
    "algo.maxwell_solver=Yee",
    "algo.evolve_scheme=explicit",
    f"algo.em_solver_medium={medium}",
    "boundary.field_lo=periodic periodic periodic",
    "boundary.field_hi=periodic periodic periodic",
    "boundary.particle_lo=periodic periodic periodic",
    "boundary.particle_hi=periodic periodic periodic",
    "particles.species_names=empty",
    "empty.species_type=electron",
    "empty.injection_style=none",
    "algo.particle_shape=1",
    "warpx.do_dive_cleaning=0",
    "warpx.do_divb_cleaning=0",
    "warpx.use_filter=0",
    "warpx.verbose=0",
    "diagnostics.enable=0",
    "warpx.const_dt=5e-13",
    "max_step=8",
    "warpx.E_ext_grid_init_style=parse_E_ext_grid_function",
    'warpx.Ex_external_grid_function(x,y,z)="2000*sin(2*pi*y/.032)"',
    'warpx.Ey_external_grid_function(x,y,z)="1300*cos(2*pi*z/.032)"',
    'warpx.Ez_external_grid_function(x,y,z)="800*sin(2*pi*x/.032)"',
]
if medium == "macroscopic":
    parameters += [
        f"algo.macroscopic_sigma_method={args.medium}",
        "macroscopic.sigma=0.0001",
        "macroscopic.epsilon=8.8541878188e-12",
        "macroscopic.mu=1.2566370612685e-6",
    ]
if args.levels == 2:
    parameters += [
        "warpx.fine_tag_lo=.008 .008 .008",
        "warpx.fine_tag_hi=.024 .024 .024",
        "warpx.pml_ncell=4",
        "warpx.pml_delta=4",
    ]
geometry.dims = 3
libwarpx.initialize([sys.executable, *parameters])
register = libwarpx.warpx.multifab_register()
sequence = []


def electric_norms():
    """Read valid cells only, including every level visible to a callback."""
    norms = []
    for level in range(args.levels):
        norm = 0.0
        for direction in "xyz":
            field = register.get("Efield_fp", dir=direction, level=level)
            for tile in field:
                valid, fab = tile.validbox(), tile.fabbox()
                values = field.array(tile).to_numpy(copy=True, order="F")
                slices = tuple(
                    slice(
                        valid.small_end[d] - fab.small_end[d],
                        valid.big_end[d] - fab.small_end[d] + 1,
                    )
                    for d in range(3)
                )
                norm += float(np.sum(values[slices] ** 2))
        norms.append(norm)
    return np.array(norms)


previous_norms = electric_norms()


def after_b():
    sequence.append("B")


def after_e():
    global previous_norms
    sequence.append("E")
    updated_norms = electric_norms()
    assert np.all(updated_norms != previous_norms), (
        "afterEpush must observe the advanced electric fields on every level"
    )
    previous_norms = updated_norms


callbacks.installafterBpush(after_b)
callbacks.installafterEpush(after_e)
libwarpx.warpx.evolve(8)
assert sequence == ["B", "E", "B"] * 8, sequence
print("afterEpush count, order and all-level field advancement passed")
libwarpx.finalize()
