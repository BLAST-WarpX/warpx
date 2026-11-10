"""Check all exterior or overlapping PML faces and corners with unequal XYZ widths."""

import argparse

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--in-domain", action="store_true")
args = parser.parse_args()
geometry.dims = 3
width = np.array([3, 4, 5])
extent = np.array([24, 24, 24])
libwarpx.initialize(
    [
        "directional-pml-faces",
        "geometry.dims=3",
        "geometry.prob_lo=0 0 0",
        "geometry.prob_hi=1 1 1",
        "amr.n_cell=24 24 24",
        "amr.max_level=0",
        "amr.max_grid_size=12",
        "amr.blocking_factor=4",
        "boundary.field_lo=pml pml pml",
        "boundary.field_hi=pml pml pml",
        "warpx.pml_ncell_vect=3 4 5",
        "warpx.pml_delta_vect=4 2 3",
        f"warpx.do_pml_in_domain={int(args.in_domain)}",
        "algo.maxwell_solver=yee",
        "warpx.verbose=0",
        "amrex.the_arena_init_size=0",
    ]
)
outer_lo = np.zeros(3, dtype=int) if args.in_domain else -width
outer_hi = extent if args.in_domain else extent + width
inner_lo = width if args.in_domain else np.zeros(3, dtype=int)
inner_hi = extent - width if args.in_domain else extent
expected = np.ones(outer_hi - outer_lo, dtype=bool)
expected[
    tuple(slice(lo, hi) for lo, hi in zip(inner_lo - outer_lo, inner_hi - outer_lo))
] = False
coverage = np.zeros_like(expected, dtype=int)
field = libwarpx.warpx.multifab_register().get("pml_B_fp", dir="x", level=0)
for mfi in field:
    box = mfi.validbox()
    lo = np.array(list(box.small_end)) - outer_lo
    # The magnetic x face is nodal in x and cell centered in y and z.
    hi = np.array(list(box.big_end)) + [0, 1, 1] - outer_lo
    assert np.all(lo >= 0) and np.all(hi <= coverage.shape)
    coverage[tuple(slice(low, high) for low, high in zip(lo, hi))] += 1
np.testing.assert_array_equal(coverage, expected)
libwarpx.finalize()
