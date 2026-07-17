"""
This file is part of WarpX

Copyright 2024 WarpX community
Authors: Axel Huebl, David Grote
License: BSD-3-Clause-LBNL
"""


def mesh(self, direction, include_ghosts=False):
    """Returns the mesh along the specified direction with the appropriate centering.

    Parameters
    ----------
    direction: string
        In 3d, one of 'x', 'y', or 'z'.
        In 2d, Cartesian, one of 'x', or 'z'.
        In RZ, one of 'r', or 'z'
        In Z, 'z'.

    include_ghosts: bool, default = False
        Whether the ghosts cells are included in the mesh
    """
    from .._libwarpx import libwarpx

    try:
        if libwarpx.geometry_dim == "3d":
            idir = ["x", "y", "z"].index(direction)
        elif libwarpx.geometry_dim == "2d":
            idir = ["x", "z"].index(direction)
        elif libwarpx.geometry_dim == "rz":
            idir = ["r", "z"].index(direction)
        elif libwarpx.geometry_dim == "1d":
            idir = ["z"].index(direction)
        else:
            raise Exception(f"Direction not implemented: {libwarpx.geometry_dim}")
    except ValueError:
        raise Exception("Inappropriate direction given")

    # Cell size and lo are obtained from warpx, the imesh from the MultiFab
    warpx = libwarpx.libwarpx_so.get_instance()
    dd = warpx.Geom(self.level).data().CellSize(idir)
    lo = warpx.Geom(self.level).ProbLo(idir)
    imesh = self.imesh(idir, include_ghosts)
    return lo + imesh * dd


def register_warpx_MultiFab_extension(amr):
    """MultiFab helper methods"""

    # register properties for the MultiFab type
    amr.MultiFab.level = None  # set by the WarpX getter

    # register member functions for the MultiFab type
    amr.MultiFab.mesh = mesh
