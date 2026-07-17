"""
This file is part of WarpX

Copyright 2024 WarpX community
Authors: Axel Huebl, David Grote
License: BSD-3-Clause-LBNL
"""


def get_particle_container_from_name(self, name):
    """Deprecated alias for get"""
    import warnings

    warnings.warn(
        "get_particle_container_from_name is deprecated. Use get instead.",
        UserWarning,
        stacklevel=2,
    )

    return self.get(name)


def register_warpx_MultiParticleContainer_extension(libwarpx_so):
    """MultiParticleContainer helper methods"""

    # Register additional methods
    libwarpx_so.WarpXParticleContainer.get_particle_container_from_name = (
        get_particle_container_from_name
    )
