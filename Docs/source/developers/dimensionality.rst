.. _developers-dimensionality:

Dimensionality
==============

This section describes the handling of dimensionality in WarpX.
The velocity space is always three-dimensional: regardless of the dimensionality of the configuration space (position space), particles always carry three velocity (or momentum) components.

Build Options
-------------

==============  ==========================
Dimensions      CMake Option
==============  ==========================
**3D**          ``WarpX_DIMS=3`` (default)
**2D**          ``WarpX_DIMS=2``
**1D**          ``WarpX_DIMS=1``
**RZ**          ``WarpX_DIMS=RZ``
**RCYLINDER**   ``WarpX_DIMS=RCYLINDER``
**RSPHERE**     ``WarpX_DIMS=RSPHERE``
==============  ==========================

Note that one can :ref:`build multiple WarpX dimensions at once <install-build-options>` via ``-DWarpX_DIMS="1;2;3;RZ;RCYLINDER;RSPHERE"``.

See :ref:`building from source <install-build-cmake>` for further details.

Defines
-------

Depending on the build variant of WarpX, the following preprocessor macros will be set:

=========================  ===========  ===========  ===========  ===========  ===========  ===========
Macro                      3D           2D           1D           RZ           RCYLINDER    RSPHERE
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
**Convention**        **3D**       **2D**       **1D**       **RZ**       **RCYLINDER**    **RSPHERE**
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

In 2D, we assume that the position of a particle in ``y`` is equal to ``0``.
In 1D, we assume that the position of a particle in ``x`` and ``y`` is equal to ``0``.

Towards Runtime Dimensionality
------------------------------

WarpX is migrating towards a single binary that supports all dimensionalities,
selected at runtime via the ``geometry.dims`` input parameter, built against
AMReX compiled once with ``AMREX_SPACEDIM=3``.
The transition infrastructure lives in ``Source/Utils/WarpXDim.H`` (the runtime
``warpx::Dim`` enum, the ``warpx::CompileDim`` bridge from the macros above and
constexpr traits like ``warpx::has_x``), ``Source/Utils/WarpXDimDispatch.H``
(``warpx::dim_dispatch``, mapping the runtime dimensionality to a compile-time
template argument like the existing particle-shape-order dispatch) and
``Source/Utils/WarpXDimIndexing.H`` (``warpx::IdxMap``, replacing
``WARPX_ZINDEX``, and ``warpx::field_at``).

Kernels are converted from ``#if defined(WARPX_DIM_*)`` blocks to
``template <warpx::Dim D>`` with ``if constexpr`` branches that default to
``warpx::CompileDim``, so per-dimension builds compile unchanged and produce
bit-identical results during the migration.

In unified builds (``WARPX_DIM_RUNTIME``), fields of all dimensionalities are
stored in degenerate-3D ``MultiFab`` s: collapsed dimensions have extent 1,
``prob_lo/hi = [-0.5, 0.5)`` (unit cell size, matching the conventions of
``WarpX::CellSize`` for absent dimensions), zero guard cells, cell-centered
staggering and periodic boundaries; ``z`` is always at index 2 (1D uses
``(1, 1, nz)``, 2D and RZ use ``(nx, 1, nz)``).
Particle positions are always three SoA components; components of collapsed
dimensions are 0 and are never updated by the pusher.
See ``Tools/Prototypes/RuntimeDims/`` for the prototype application that runs
1D, 2D and 3D simulations from one executable.
