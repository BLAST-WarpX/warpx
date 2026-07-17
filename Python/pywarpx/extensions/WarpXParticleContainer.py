"""
This file is part of WarpX

Copyright 2024 WarpX community
Authors: Axel Huebl, David Grote
License: BSD-3-Clause-LBNL
"""


def add_particles(
    self,
    x=None,
    y=None,
    z=None,
    ux=None,
    uy=None,
    uz=None,
    w=None,
    unique_particles=True,
    **kwargs,
):
    """
    A function for adding particles to the WarpX simulation.

    Parameters
    ----------

    species_name     : str
        The type of species for which particles will be added

    x, y, z          : arrays or scalars
        The particle positions (m) (default = 0.)

    ux, uy, uz       : arrays or scalars
        The particle proper velocities (m/s) (default = 0.)

    w                : array or scalars
        Particle weights (default = 0.)

    unique_particles : bool
        True means the added particles are duplicated by each process;
        False means the number of added particles is independent of
        the number of processes (default = True)

    kwargs           : dict
        Containing an entry for all the extra particle attribute arrays. If
        an attribute is not given it will be set to 0.
    """
    import numpy as np

    from .._libwarpx import libwarpx

    # --- Get length of arrays, set to one for scalars
    lenx = np.size(x)
    leny = np.size(y)
    lenz = np.size(z)
    lenux = np.size(ux)
    lenuy = np.size(uy)
    lenuz = np.size(uz)
    lenw = np.size(w)

    # --- Find the max length of the parameters supplied
    maxlen = 0
    if x is not None:
        maxlen = max(maxlen, lenx)
    if y is not None:
        maxlen = max(maxlen, leny)
    if z is not None:
        maxlen = max(maxlen, lenz)
    if ux is not None:
        maxlen = max(maxlen, lenux)
    if uy is not None:
        maxlen = max(maxlen, lenuy)
    if uz is not None:
        maxlen = max(maxlen, lenuz)
    if w is not None:
        maxlen = max(maxlen, lenw)

    # --- Make sure that the lengths of the input parameters are consistent
    assert x is None or lenx == maxlen or lenx == 1, (
        "Length of x doesn't match len of others"
    )
    assert y is None or leny == maxlen or leny == 1, (
        "Length of y doesn't match len of others"
    )
    assert z is None or lenz == maxlen or lenz == 1, (
        "Length of z doesn't match len of others"
    )
    assert ux is None or lenux == maxlen or lenux == 1, (
        "Length of ux doesn't match len of others"
    )
    assert uy is None or lenuy == maxlen or lenuy == 1, (
        "Length of uy doesn't match len of others"
    )
    assert uz is None or lenuz == maxlen or lenuz == 1, (
        "Length of uz doesn't match len of others"
    )
    assert w is None or lenw == maxlen or lenw == 1, (
        "Length of w doesn't match len of others"
    )
    for key, val in kwargs.items():
        assert np.size(val) == 1 or len(val) == maxlen, (
            f"Length of {key} doesn't match len of others"
        )

    # --- Broadcast scalars into appropriate length arrays
    # --- If the parameter was not supplied, use the default value
    if lenx == 1:
        x = np.full(maxlen, (x or 0.0))
    if leny == 1:
        y = np.full(maxlen, (y or 0.0))
    if lenz == 1:
        z = np.full(maxlen, (z or 0.0))
    if lenux == 1:
        ux = np.full(maxlen, (ux or 0.0))
    if lenuy == 1:
        uy = np.full(maxlen, (uy or 0.0))
    if lenuz == 1:
        uz = np.full(maxlen, (uz or 0.0))
    if lenw == 1:
        w = np.full(maxlen, (w or 0.0))
    for key, val in kwargs.items():
        if np.size(val) == 1:
            kwargs[key] = np.full(maxlen, val)

    # --- The number of built in attributes
    # --- The positions
    built_in_attrs = libwarpx.dim
    # --- The three velocities
    built_in_attrs += 3
    if libwarpx.geometry_dim == "rz":
        # --- With RZ, there is also theta
        built_in_attrs += 1

    # --- The number of extra attributes (including the weight)
    nattr = self.num_real_comps - built_in_attrs
    attr = np.zeros((maxlen, nattr))
    attr[:, 0] = w

    # --- Note that the velocities are handled separately and not included in attr
    # --- (even though they are stored as attributes in the C++)
    for key, vals in kwargs.items():
        attr[:, self.get_real_comp_index(key) - built_in_attrs] = vals

    nattr_int = 0
    attr_int = np.empty([0], dtype=np.int32)

    # TODO: expose ParticleReal through pyAMReX
    # and cast arrays to the correct types, before calling add_n_particles
    # x = x.astype(self._numpy_particlereal_dtype, copy=False)
    # y = y.astype(self._numpy_particlereal_dtype, copy=False)
    # z = z.astype(self._numpy_particlereal_dtype, copy=False)
    # ux = ux.astype(self._numpy_particlereal_dtype, copy=False)
    # uy = uy.astype(self._numpy_particlereal_dtype, copy=False)
    # uz = uz.astype(self._numpy_particlereal_dtype, copy=False)

    self.add_n_particles(
        0,
        x.size,
        x,
        y,
        z,
        ux,
        uy,
        uz,
        nattr,
        attr,
        nattr_int,
        attr_int,
        unique_particles,
    )


def register_warpx_WarpXParticleContainer_extension(libwarpx_so):
    """WarpXParticleContainer helper methods"""

    # Register the overload dispatcher
    #   note: this currently overwrites the pyAMReX signature
    #         add_particles(other: ParticleContainer, local: bool = False)
    libwarpx_so.WarpXParticleContainer.add_particles = add_particles
