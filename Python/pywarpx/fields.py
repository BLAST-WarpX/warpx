# Copyright 2017-2023 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Provides wrappers around multiFABs

Available routines:

ExWrapper, EyWrapper, EzWrapper
BxWrapper, ByWrapper, BzWrapper
JxWrapper, JyWrapper, JzWrapper

ExFPWrapper, EyFPWrapper, EzFPWrapper
BxFPWrapper, ByFPWrapper, BzFPWrapper
JxFPWrapper, JyFPWrapper, JzFPWrapper

RhoFPWrapper, PhiFPWrapper
FFPWrapper, GFPWrapper
AxFPWrapper, AyFPWrapper, AzFPWrapper

ExCPWrapper, EyCPWrapper, EzCPWrapper
BxCPWrapper, ByCPWrapper, BzCPWrapper
JxCPWrapper, JyCPWrapper, JzCPWrapper

RhoCPWrapper
FCPWrapper, GCPWrapper

EdgeLengthsxWrapper, EdgeLengthsyWrapper, EdgeLengthszWrapper
FaceAreasxWrapper, FaceAreasyWrapper, FaceAreaszWrapper

ExFPPMLWrapper, EyFPPMLWrapper, EzFPPMLWrapper
BxFPPMLWrapper, ByFPPMLWrapper, BzFPPMLWrapper
JxFPPMLWrapper, JyFPPMLWrapper, JzFPPMLWrapper
JxFPPlasmaWrapper, JyFPPlasmaWrapper, JzFPPlasmaWrapper
FFPPMLWrapper, GFPPMLWrapper

ExCPPMLWrapper, EyCPPMLWrapper, EzCPPMLWrapper
BxCPPMLWrapper, ByCPPMLWrapper, BzCPPMLWrapper
JxCPPMLWrapper, JyCPPMLWrapper, JzCPPMLWrapper
FCPPMLWrapper, GCPPMLWrapper
"""

import warnings

from ._libwarpx import libwarpx


class _MultiFABWrapper(object):
    """Wrapper around MultiFabs
    This provides a convenient way to query and set data in the MultiFabs.
    The indexing is based on global indices.

    Parameters
    ----------
     mf: MultiFab
         The Multifab that is wrapped.

     mf_name: string
         The name of the MultiFab to be accessed, the tag specified when the
         MultiFab is allocated. The Multifab will be accessed anew from WarpX
         everytime it is called if this argument is given instead of directly
         providing the Multifab.

     idir: int, optional
         For MultiFab that is an element of a vector, the direction number, 0, 1, or 2.

     level: int
         The refinement level
    """

    def __init__(self, mf=None, mf_name=None, idir=None, level=0, include_ghosts=None):
        self._mf = mf
        self.mf_name = mf_name
        self.idir = idir
        self.level = level

        self.dim = libwarpx.dim

        if include_ghosts is not None:
            warnings.warn(
                "The include_ghosts argument is deprecated. Ghost cells are accessed using imaginary numbers, negative for the lower, positive for the upper."
            )

    def __len__(self):
        "Returns the number of blocks"
        return self.mf.size

    def __iter__(self):
        "The iteration is over the MultiFab"
        return self.mf.__iter__()

    def __getitem__(self, index):
        """Returns slice of the MultiFab using global indexing, as a numpy array.
        This uses numpy array indexing, with the indexing relative to the global array.
        The slice ranges can cross multiple blocks and the result will be gathered into a single
        array.

        In an MPI context, this is a global operation. An "allgather" is performed so that the full
        result is returned on all processors.

        Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

        The default range of the indices includes only the valid cells. The ":" index will include all of
        the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
        negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
        The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
        ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

        Parameters
        ----------
        index : the index using numpy style indexing
            Index of the slice to return.
        """
        return self.mf.__getitem__(index)

    def __setitem__(self, index, value):
        """Sets the slice of the MultiFab using global indexing.
        This uses numpy array indexing, with the indexing relative to the global array.
        The slice ranges can cross multiple blocks and the value will be distributed accordingly.
        Note that this will apply the value to both valid and ghost cells.

        In an MPI context, this is a local operation. On each processor, the blocks within the slice
        range will be set to the value.

        Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

        The default range of the indices includes only the valid cells. The ":" index will include all of
        the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
        negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
        The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
        ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

        Parameters
        ----------
        index : the index using numpy style indexing
            Index of the slice to return.
        value : scalar or array
            Input value to assign to the specified slice of the MultiFab
        """
        self.mf.__setitem__(index, value)

    def __getattr__(self, name):
        return getattr(self.mf, name)

    @property
    def mf(self):
        if self._mf is not None:
            return self._mf
        else:
            # Always fetch this anew in case the C++ MultiFab is recreated
            warpx = libwarpx.libwarpx_so.get_instance()
            if self.idir is not None:
                direction = libwarpx.libwarpx_so.Direction(self.idir)
                return warpx.multifab(self.mf_name, direction, self.level)
            else:
                return warpx.multifab(self.mf_name, self.level)

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

        try:
            if libwarpx.geometry_dim == "3d":
                idir = ["x", "y", "z"].index(direction)
            elif libwarpx.geometry_dim == "2d":
                idir = ["x", "z"].index(direction)
            elif libwarpx.geometry_dim == "rz":
                idir = ["r", "z"].index(direction)
            elif libwarpx.geometry_dim == "1d":
                idir = ["z"].index(direction)
        except ValueError:
            raise Exception("Inappropriate direction given")

        # Cell size and lo are obtained from warpx, the imesh from the MultiFab
        warpx = libwarpx.libwarpx_so.get_instance()
        dd = warpx.Geom(self.level).data().CellSize(idir)
        lo = warpx.Geom(self.level).ProbLo(idir)
        imesh = self.mf.imesh(idir, include_ghosts)
        return lo + imesh * dd


def CustomNamedxWrapper(mf_name, level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name=mf_name, idir=0, level=level, include_ghosts=include_ghosts
    )


def CustomNamedyWrapper(mf_name, level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name=mf_name, idir=1, level=level, include_ghosts=include_ghosts
    )


def CustomNamedzWrapper(mf_name, level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name=mf_name, idir=2, level=level, include_ghosts=include_ghosts
    )


def ExWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_aux", idir=0, level=level, include_ghosts=include_ghosts
    )


def EyWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_aux", idir=1, level=level, include_ghosts=include_ghosts
    )


def EzWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_aux", idir=2, level=level, include_ghosts=include_ghosts
    )


def BxWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_aux", idir=0, level=level, include_ghosts=include_ghosts
    )


def ByWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_aux", idir=1, level=level, include_ghosts=include_ghosts
    )


def BzWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_aux", idir=2, level=level, include_ghosts=include_ghosts
    )


def JxWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_fp", idir=0, level=level, include_ghosts=include_ghosts
    )


def JyWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_fp", idir=1, level=level, include_ghosts=include_ghosts
    )


def JzWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_fp", idir=2, level=level, include_ghosts=include_ghosts
    )


def ExFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_fp", idir=0, level=level, include_ghosts=include_ghosts
    )


def EyFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_fp", idir=1, level=level, include_ghosts=include_ghosts
    )


def EzFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_fp", idir=2, level=level, include_ghosts=include_ghosts
    )


def BxFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_fp", idir=0, level=level, include_ghosts=include_ghosts
    )


def ByFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_fp", idir=1, level=level, include_ghosts=include_ghosts
    )


def BzFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_fp", idir=2, level=level, include_ghosts=include_ghosts
    )


def ExFPExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_fp_external", idir=0, level=level, include_ghosts=include_ghosts
    )


def EyFPExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_fp_external", idir=1, level=level, include_ghosts=include_ghosts
    )


def EzFPExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_fp_external", idir=2, level=level, include_ghosts=include_ghosts
    )


def BxFPExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_fp_external", idir=0, level=level, include_ghosts=include_ghosts
    )


def ByFPExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_fp_external", idir=1, level=level, include_ghosts=include_ghosts
    )


def BzFPExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_fp_external", idir=2, level=level, include_ghosts=include_ghosts
    )


def AxHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_A_fp_external",
        idir=0,
        level=level,
        include_ghosts=include_ghosts,
    )


def AyHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_A_fp_external",
        idir=1,
        level=level,
        include_ghosts=include_ghosts,
    )


def AzHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_A_fp_external",
        idir=2,
        level=level,
        include_ghosts=include_ghosts,
    )


def ExHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_E_fp_external",
        idir=0,
        level=level,
        include_ghosts=include_ghosts,
    )


def EyHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_E_fp_external",
        idir=1,
        level=level,
        include_ghosts=include_ghosts,
    )


def EzHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_E_fp_external",
        idir=2,
        level=level,
        include_ghosts=include_ghosts,
    )


def BxHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_B_fp_external",
        idir=0,
        level=level,
        include_ghosts=include_ghosts,
    )


def ByHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_B_fp_external",
        idir=1,
        level=level,
        include_ghosts=include_ghosts,
    )


def BzHybridExternalWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_B_fp_external",
        idir=2,
        level=level,
        include_ghosts=include_ghosts,
    )


def JxFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_fp", idir=0, level=level, include_ghosts=include_ghosts
    )


def JyFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_fp", idir=1, level=level, include_ghosts=include_ghosts
    )


def JzFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_fp", idir=2, level=level, include_ghosts=include_ghosts
    )


def RhoFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="rho_fp", level=level, include_ghosts=include_ghosts
    )


def PhiFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="phi_fp", level=level, include_ghosts=include_ghosts
    )


def FFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(mf_name="F_fp", level=level, include_ghosts=include_ghosts)


def GFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(mf_name="G_fp", level=level, include_ghosts=include_ghosts)


def AxFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="vector_potential_fp_nodal",
        idir=0,
        level=level,
        include_ghosts=include_ghosts,
    )


def AyFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="vector_potential_fp_nodal",
        idir=1,
        level=level,
        include_ghosts=include_ghosts,
    )


def AzFPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="vector_potential_fp_nodal",
        idir=2,
        level=level,
        include_ghosts=include_ghosts,
    )


def ExCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_cp", idir=0, level=level, include_ghosts=include_ghosts
    )


def EyCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_cp", idir=1, level=level, include_ghosts=include_ghosts
    )


def EzCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Efield_cp", idir=2, level=level, include_ghosts=include_ghosts
    )


def BxCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_cp", idir=0, level=level, include_ghosts=include_ghosts
    )


def ByCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_cp", idir=1, level=level, include_ghosts=include_ghosts
    )


def BzCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="Bfield_cp", idir=2, level=level, include_ghosts=include_ghosts
    )


def JxCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_cp", idir=0, level=level, include_ghosts=include_ghosts
    )


def JyCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_cp", idir=1, level=level, include_ghosts=include_ghosts
    )


def JzCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="current_cp", idir=2, level=level, include_ghosts=include_ghosts
    )


def RhoCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="rho_cp", level=level, include_ghosts=include_ghosts
    )


def FCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(mf_name="F_cp", level=level, include_ghosts=include_ghosts)


def GCPWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(mf_name="G_cp", level=level, include_ghosts=include_ghosts)


def EdgeLengthsxWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="edge_lengths", idir=0, level=level, include_ghosts=include_ghosts
    )


def EdgeLengthsyWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="edge_lengths", idir=1, level=level, include_ghosts=include_ghosts
    )


def EdgeLengthszWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="edge_lengths", idir=2, level=level, include_ghosts=include_ghosts
    )


def FaceAreasxWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="face_areas", idir=0, level=level, include_ghosts=include_ghosts
    )


def FaceAreasyWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="face_areas", idir=1, level=level, include_ghosts=include_ghosts
    )


def FaceAreaszWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="face_areas", idir=2, level=level, include_ghosts=include_ghosts
    )


def JxFPPlasmaWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_current_fp_plasma",
        idir=0,
        level=level,
        include_ghosts=include_ghosts,
    )


def JyFPPlasmaWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_current_fp_plasma",
        idir=1,
        level=level,
        include_ghosts=include_ghosts,
    )


def JzFPPlasmaWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="hybrid_current_fp_plasma",
        idir=2,
        level=level,
        include_ghosts=include_ghosts,
    )


def ExFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_E_fp", idir=0, level=level, include_ghosts=include_ghosts
    )


def EyFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_E_fp", idir=1, level=level, include_ghosts=include_ghosts
    )


def EzFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_E_fp", idir=2, level=level, include_ghosts=include_ghosts
    )


def BxFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_B_fp", idir=0, level=level, include_ghosts=include_ghosts
    )


def ByFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_B_fp", idir=1, level=level, include_ghosts=include_ghosts
    )


def BzFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_B_fp", idir=2, level=level, include_ghosts=include_ghosts
    )


def JxFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_j_fp", idir=0, level=level, include_ghosts=include_ghosts
    )


def JyFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_j_fp", idir=1, level=level, include_ghosts=include_ghosts
    )


def JzFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_j_fp", idir=2, level=level, include_ghosts=include_ghosts
    )


def FFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_F_fp", level=level, include_ghosts=include_ghosts
    )


def GFPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_G_fp", level=level, include_ghosts=include_ghosts
    )


def ExCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_E_cp", idir=0, level=level, include_ghosts=include_ghosts
    )


def EyCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_E_cp", idir=1, level=level, include_ghosts=include_ghosts
    )


def EzCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_E_cp", idir=2, level=level, include_ghosts=include_ghosts
    )


def BxCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_B_cp", idir=0, level=level, include_ghosts=include_ghosts
    )


def ByCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_B_cp", idir=1, level=level, include_ghosts=include_ghosts
    )


def BzCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_B_cp", idir=2, level=level, include_ghosts=include_ghosts
    )


def JxCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_j_cp", idir=0, level=level, include_ghosts=include_ghosts
    )


def JyCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_j_cp", idir=1, level=level, include_ghosts=include_ghosts
    )


def JzCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_j_cp", idir=2, level=level, include_ghosts=include_ghosts
    )


def FCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_F_cp", level=level, include_ghosts=include_ghosts
    )


def GCPPMLWrapper(level=0, include_ghosts=None):
    return _MultiFABWrapper(
        mf_name="pml_G_cp", level=level, include_ghosts=include_ghosts
    )
