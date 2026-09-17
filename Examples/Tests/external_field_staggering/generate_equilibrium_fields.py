#!/usr/bin/env python3
"""Write the screw-pinch equilibrium to openPMD files for the external-field reader.

Two variants of the same fields:
  ext_fields_nodal.h5 : all components sampled on the nodal lattice
                        (position = [0,0,0]); the reader must resample
                        them onto the Yee grid with its interpolation.
  ext_fields_yee.h5   : each component sampled at its Yee position
                        (position attribute set accordingly); with the
                        position fix the reader samples these exactly.

The file lattice matches the WarpX grid spacing and node locations,
padded by NPAD cells on every side so that staggered target points near
the domain boundary stay strictly inside the file domain.
"""

import sys

import numpy as np
import openpmd_api as io
from equilibrium import (
    FIELD_FUNCS,
    NCELL,
    NODAL_POSITION,
    NPAD,
    PROB_LO,
    YEE_POSITION,
    H,
    check_pressure_balance,
)

UNIT_DIM = {
    # V/m = kg m / (A s^3)
    "E": {
        io.Unit_Dimension.L: 1,
        io.Unit_Dimension.M: 1,
        io.Unit_Dimension.T: -3,
        io.Unit_Dimension.I: -1,
    },
    # T = kg / (A s^2)
    "B": {io.Unit_Dimension.M: 1, io.Unit_Dimension.T: -2, io.Unit_Dimension.I: -1},
}


def write_file(fname, positions):
    file_lo = PROB_LO - NPAD * H
    nnode = NCELL + 1 + 2 * NPAD

    series = io.Series(fname, io.Access.create)
    it = series.iterations[0]
    it.time = 0.0
    it.dt = 1.0

    for fnamekey in ("E", "B"):
        mesh = it.meshes[fnamekey]
        mesh.geometry = io.Geometry.cartesian
        mesh.data_order = "C"
        mesh.axis_labels = ["x", "y", "z"]
        mesh.grid_spacing = list(H)
        mesh.grid_global_offset = list(file_lo)
        mesh.grid_unit_SI = 1.0
        mesh.unit_dimension = UNIT_DIM[fnamekey]

        for comp in ("x", "y", "z"):
            cname = fnamekey + comp
            pos = positions[cname]
            coords = [
                file_lo[d] + (np.arange(nnode[d]) + pos[d]) * H[d] for d in range(3)
            ]
            X, Y, Z = np.meshgrid(*coords, indexing="ij")
            data = np.ascontiguousarray(FIELD_FUNCS[cname](X, Y, Z), dtype=np.float64)

            rc = mesh[comp]
            rc.position = list(pos)
            rc.unit_SI = 1.0
            rc.reset_dataset(io.Dataset(data.dtype, data.shape))
            rc.store_chunk(data)
            series.flush()

    series.close()
    print(f"wrote {fname}  (lattice {nnode.tolist()} nodes, offset {file_lo.tolist()})")


def main(outdir="."):
    res = check_pressure_balance()
    print(f"pressure balance residual (normalized by p0/a): {res:.3e}")
    write_file(f"{outdir}/ext_fields_nodal.h5", NODAL_POSITION)
    write_file(f"{outdir}/ext_fields_yee.h5", YEE_POSITION)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
