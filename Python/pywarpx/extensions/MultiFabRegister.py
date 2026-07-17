"""
This file is part of WarpX

Copyright 2025 WarpX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def get(self, name, dir=None, level=None):
    """Return a MultiFab that is part of a vector/tensor field.

    This throws a runtime error if the requested field is not present.

    Parameters
    ----------
    self : MultiFabRegister
        The MultiFabRegister.
    name : str
        The name of the field
    dir : int or None, optional
        The field component for vector fields ("direction" of the unit vector).
        Required for vector/tensor fields, omitted for scalar fields.
    level : int
        The MR level.

    Returns
    -------
    MultiFab
        A non-owning reference to the MultiFab (field)
    """
    if level is None:
        raise ValueError("level must be specified")

    if dir is None:
        mf = self._get(name=name, level=level)
    else:
        mf = self._get(name=name, dir=dir, level=level)

    mf.level = level
    return mf


def register_warpx_MultiFabRegister_extension(libwarpx_so):
    """MultiFabRegister helper methods"""

    # register member functions for the MultiFabRegister type
    libwarpx_so.MultiFabRegister.get = get
