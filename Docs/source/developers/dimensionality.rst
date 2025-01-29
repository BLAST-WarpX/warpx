.. _developers-dimensionality:

Dimensionality
==============

This section describes the handling of dimensionality in WarpX.

Build Options
-------------

==========  ==========================
Dimensions  CMake Option
==========  ==========================
**3D3V**    ``WarpX_DIMS=3`` (default)
**2D3V**    ``WarpX_DIMS=2``
**1D3V**    ``WarpX_DIMS=1``
**RZ**      ``WarpX_DIMS=RZ``
==========  ==========================

Note that one can :ref:`build multiple WarpX dimensions at once <building-cmake-options>` via ``-DWarpX_DIMS="1;2;RZ;3"``.

See :ref:`building from source <install-developers>` for further details.

Defines
-------

Depending on the build variant of WarpX, the following preprocessor macros will be set:

==================  ===========  ===========  ===========  ===========
Macro               3D3V         2D3V         1D3V         RZ
==================  ===========  ===========  ===========  ===========
``AMREX_SPACEDIM``  ``3``        ``2``        ``1``        ``2``
``WARPX_DIM_3D``    **defined**  *undefined*  *undefined*  *undefined*
``WARPX_DIM_1D_Z``  *undefined*  *undefined*  **defined**  *undefined*
``WARPX_DIM_XZ``    *undefined*  **defined**  *undefined*  *undefined*
``WARPX_DIM_RZ``    *undefined*  *undefined*  *undefined*  **defined**
``WARPX_ZINDEX``    ``2``        ``1``        ``0``        ``1``
==================  ===========  ===========  ===========  ===========

At the same time, the following conventions will apply:

====================  ===========  ===========  ===========  ===========
**Convention**        **3D3V**     **2D3V**     **1D3V**     **RZ**
--------------------  -----------  -----------  -----------  -----------
*Fields*
------------------------------------------------------------------------
AMReX Box dimensions  ``3``         ``2``       ``1``        ``2``
WarpX axis labels     ``x, y, z``   ``x, z``    ``z``        ``x, z``
--------------------  -----------  -----------  -----------  -----------
*Particles*
------------------------------------------------------------------------
AMReX ``.pos()``      ``0, 1, 2``  ``0, 1``     ``0``        ``0, 1``
WarpX position names  ``x, y, z``  ``x, z``     ``z``        ``r, z``
extra SoA attribute                                          ``theta``
====================  ===========  ===========  ===========  ===========

Please see the following sections for particle SoA details.

Conventions
-----------

In 2D3V, we assume that the position of a particle in ``y`` is equal to ``0``.
In 1D3V, we assume that the position of a particle in ``x`` and ``y`` is equal to ``0``.
