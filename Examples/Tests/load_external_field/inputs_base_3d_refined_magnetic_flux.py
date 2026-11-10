#!/usr/bin/env python3
"""Check coarse face flux and fine-cell divergence with transverse Yee refinement."""

import argparse

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type=int, nargs=3, required=True)
ratio = np.array(parser.parse_args().ratio)
geometry.dims = 3
# The affine field has zero curl and divergence. Initialize only level zero so
# every nonzero fine auxiliary value is a prolongated magnetic correction.
libwarpx.initialize(
    [
        "refined-magnetic-flux",
        "geometry.dims=3",
        "geometry.prob_lo=0 0 0",
        "geometry.prob_hi=1 1 1",
        "amr.n_cell=32 32 32",
        "amr.max_grid_size=16",
        "amr.blocking_factor=4",
        "amr.n_error_buf=0",
        "amr.max_level=1",
        "amr.ref_ratio_vect=" + " ".join(map(str, ratio)),
        "warpx.fine_tag_lo=0.25 0.25 0.25",
        "warpx.fine_tag_hi=0.75 0.75 0.75",
        "boundary.field_lo=periodic periodic periodic",
        "boundary.field_hi=periodic periodic periodic",
        "warpx.eb_implicit_function=0.035^2-(x-0.1)^2-(y-0.1)^2-(z-0.1)^2",
        "warpx.build_eb_data_per_level=1",
        "warpx.maxlevel_extEMfield_init=0",
        "warpx.B_ext_grid_init_style=parse_B_ext_grid_function",
        "warpx.Bx_external_grid_function(x,y,z)=1+x+2*y",
        "warpx.By_external_grid_function(x,y,z)=2*x-2*y+z",
        "warpx.Bz_external_grid_function(x,y,z)=3+y+z",
        "warpx.do_initial_div_cleaning=0",
        "warpx.pml_ncell=4",
        "warpx.use_filter=0",
        "algo.maxwell_solver=yee",
        "algo.field_gathering=energy-conserving",
        "warpx.cfl=0.1",
        "diagnostics.enable=0",
        "amrex.the_arena_init_size=0",
        "warpx.verbose=0",
    ]
)
libwarpx.warpx.evolve(1)
register = libwarpx.warpx.multifab_register()
fine_divergence = 0
for axis, direction in enumerate("xyz"):
    field = register.get("Bfield_aux", dir=direction, level=1)
    values = np.squeeze(np.asarray(field[...]))
    fine_divergence += 32 * ratio[axis] * np.diff(values, axis=axis)
    # Sample coarse-aligned normal faces, then average all subfaces
    # transversely to recover the flux density on each coarse face.
    faces = np.take(values, np.arange(0, values.shape[axis], ratio[axis]), axis=axis)
    transverse = ratio.copy()
    transverse[axis] = 1
    shape = np.array(faces.shape) // transverse
    average = faces.reshape(tuple(np.column_stack((shape, transverse)).ravel())).mean(
        axis=(1, 3, 5)
    )
    lo = np.array(field.box_array().minimal_box().small_end) // ratio
    coords = [(lo[d] + np.arange(shape[d]) + 0.5 * (d != axis)) / 32 for d in range(3)]
    x, y, z = np.meshgrid(*coords, indexing="ij", sparse=True)
    expected = (1 + x + 2 * y, 2 * x - 2 * y + z, 3 + y + z)[axis]
    np.testing.assert_allclose(
        average, np.broadcast_to(expected, shape), rtol=2e-6, atol=1e-6
    )

np.testing.assert_allclose(fine_divergence, 0, rtol=0, atol=2e-5)
libwarpx.finalize()
