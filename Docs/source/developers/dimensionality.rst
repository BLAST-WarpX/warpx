.. _developers-dimensionality:

Dimensionality
==============

This section describes the handling of dimensionality in WarpX.

Build Options
-------------

===============  ==========================
Dimensions       CMake Option
===============  ==========================
**3D3V**         ``WarpX_DIMS=3`` (default)
**2D3V**         ``WarpX_DIMS=2``
**1D3V**         ``WarpX_DIMS=1``
**RZ3V**         ``WarpX_DIMS=RZ``
**RCYLINDER3V**  ``WarpX_DIMS=RCYLINDER``
**RSPHERE3V**    ``WarpX_DIMS=RSPHERE``
===============  ==========================

Note that one can :ref:`build multiple WarpX dimensions at once <building-cmake-options>` via ``-DWarpX_DIMS="1;2;3;RZ;RCYLINDER;RSPHERE"``.

See :ref:`building from source <install-developers>` for further details.

Defines
-------

Depending on the build variant of WarpX, the following preprocessor macros will be set:

=========================  ===========  ===========  ===========  ===========  ===========  ===========
Macro                      3D3V         2D3V         1D3V         RZ3V         RCYLINDER3V  RSPHERE3V
=========================  ===========  ===========  ===========  ===========  ===========  ===========
``AMREX_SPACEDIM``         ``3``        ``2``        ``1``        ``2``        ``1``        ``1``
``WARPX_DIM_3D``           **defined**  *undefined*  *undefined*  *undefined*  *undefined*  *undefined*
``WARPX_DIM_1D_Z``         *undefined*  *undefined*  **defined**  *undefined*  *undefined*  *undefined*
``WARPX_DIM_XZ``           *undefined*  **defined**  *undefined*  *undefined*  *undefined*  *undefined*
``WARPX_DIM_RZ``           *undefined*  *undefined*  *undefined*  **defined**  *undefined*  *undefined*
``WARPX_DIM_RCYLINDER``    *undefined*  *undefined*  *undefined*  *undefined*  **defined**  *undefined*
``WARPX_DIM_RSPHERE``      *undefined*  *undefined*  *undefined*  *undefined*  *undefined*  **defined**
``WARPX_ZINDEX``           ``2``        ``1``        ``0``        ``1``        *undefined*  *undefined*
=========================  ===========  ===========  ===========  ===========  ===========  ===========

At the same time, the following conventions will apply:

====================  ===========  ===========  ===========  ===========  ===============  ==============
**Convention**        **3D3V**     **2D3V**     **1D3V**     **RZ3V**     **RCYLINDER3V**  **RSPHERE3V**
--------------------  -----------  -----------  -----------  -----------  ---------------  --------------
*Fields*
------------------------------------------------------------------------  ---------------  --------------
AMReX Box dimensions  ``3``         ``2``       ``1``        ``2``        ``1``            ``1``
WarpX axis labels     ``x, y, z``   ``x, z``    ``z``        ``x, z``     ``r``            ``r``
--------------------  -----------  -----------  -----------  -----------  ---------------  --------------
*Particles*
------------------------------------------------------------------------  ---------------  --------------
AMReX ``.pos()``      ``0, 1, 2``  ``0, 1``     ``0``        ``0, 1``     ``0``            ``0``
WarpX position names  ``x, y, z``  ``x, z``     ``z``        ``r, z``     ``r``            ``r``
extra SoA attribute                                          ``theta``    ``theta``        ``theta, phi``
====================  ===========  ===========  ===========  ===========  ===============  ==============

Please see the following sections for particle SoA details.

Conventions
-----------

In 2D3V, we assume that the position of a particle in ``y`` is equal to ``0``.
In 1D3V, we assume that the position of a particle in ``x`` and ``y`` is equal to ``0``.
