# Copyright 2023-2025 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Roelof Groenewald (TAE Technologies)
#          S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

from ._libwarpx import libwarpx
from .Bucket import Bucket

hybridpicmodel = Bucket("hybrid_pic_model")
external_vector_potential = Bucket("external_vector_potential")


class HybridPICModel(object):
    """HybridPICModel

    Description: This is an accessor object to be able to vary parameters
    in the WarpX HybridPICModel class. This can be useful for adjusting parameters
    like the density floor or number of substeps taken dynamically. This will be implemented
    as a selection of parameters that can be adjusted at runtime in installed callbacks.
    """

    def __init__(self):
        # Nothing to be done since this is a wrapper object
        pass

    @property
    def substeps(self):
        warpx = libwarpx.libwarpx_so.get_instance()
        return warpx.get_hybrid_pic_substeps()

    @substeps.setter
    def substeps(self, substeps):
        warpx = libwarpx.libwarpx_so.get_instance()
        return warpx.set_hybrid_pic_substeps(substeps)

    @property
    def density_floor(self):
        warpx = libwarpx.libwarpx_so.get_instance()
        return warpx.get_hybrid_pic_density_floor()

    @density_floor.setter
    def density_floor(self, n_floor):
        warpx = libwarpx.libwarpx_so.get_instance()
        return warpx.set_hybrid_pic_density_floor(n_floor)
