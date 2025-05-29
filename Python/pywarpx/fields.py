# Copyright 2017-2023 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Provides wrappers around MultiFabs

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

from ._libwarpx import libwarpx


class MultiFabWrapper(object):
    """Wrapper around MultiFabs
    This provides a convenient way to query and set data in the MultiFabs.
    The indexing is based on global indices.

    Parameters
    ----------
     mf_name: string, optional
         The name of the MultiFab to be accessed, as specified in the MultiFab registry.
         The Multifab will be accessed anew from WarpX everytime it is called if this
         argument is given instead of directly providing the Multifab. Either this
         or the mf must be specified.

     mf: MultiFab, optional
         The Multifab that is wrapped. Either this or mf_name must be specified.

     idir: int, optional
         For MultiFab that is an element of a vector, the direction number, 0, 1, or 2.

     level: int
         The refinement level

     create_new: boolean, optional
         If True, a new MultiFab with the name mf_name, idir, and level will be
         created and added to the registry. The following input parameters
         are used to set the properties of the MultiFab.

     ba: BoxArray, optional
         The BoxArray for the new MultiFab, defaults to warpx.boxArray(level).

     dm: DistributionMapping, optional
         The DistributionMapping for the new MultiFab, defaults to warpx.DistributionMap(level)

     ncomp: int, optional
         The number of components for the new MultiFab, defaults to 1.

     ngrow: IntVect or int, optional
         The number of guard cells for the new MultiFab, defaults to 0.

     initial_value: float, optional
         The initial value for the new MultiFab, defaults to 0.

     remake: boolean, optional
         Whether the new MultiFab is to be remade, for example during a load balance.
         Defaults to True.

     redistribute_on_remake: boolean, optional
         Whether the data in the new MultiFab is redistributed when it is remade.
         Defaults to True.
    """

    def __init__(
        self,
        mf=None,
        mf_name=None,
        idir=None,
        level=0,
        create_new=False,
        ba=None,
        dm=None,
        ncomp=1,
        ngrow=0,
        initial_value=0.0,
        remake=True,
        redistribute_on_remake=True,
    ):
        self._mf = mf
        self.mf_name = mf_name
        self.idir = idir
        self.level = level
        self.create_new = create_new
        self.ba = ba
        self.dm = dm
        self.ncomp = ncomp
        self.ngrow = ngrow
        self.initial_value = initial_value
        self.remake = remake
        self.redistribute_on_remake = redistribute_on_remake

        self.dim = libwarpx.dim

        if self.create_new:
            self.create_new_MultiFab()

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
        # For attributes not explicitly defined here, return the
        # attribute of the underlying MultiFab
        return getattr(self.mf, name)

    @property
    def mf(self):
        if self._mf is not None:
            return self._mf
        else:
            # Always fetch this anew in case the C++ MultiFab is recreated
            warpx = libwarpx.libwarpx_so.get_instance()
            fields = warpx.multifab_register()
            if self.idir is not None:
                direction = libwarpx.libwarpx_so.Direction(self.idir)
                return fields.get(self.mf_name, direction, self.level)
            else:
                return fields.get(self.mf_name, self.level)

    def create_new_MultiFab(self):
        warpx = libwarpx.libwarpx_so.get_instance()
        fields = warpx.multifab_register()

        if self.idir is not None:
            fields.alloc_init(
                self.mf_name,
                libwarpx.libwarpx_so.Direction(self.idir),
                self.level,
                (self.ba or warpx.boxArray(self.level)),
                (self.dm or warpx.DistributionMap(self.level)),
                self.ncomp,
                libwarpx.amr.IntVect(self.ngrow),
                self.initial_value,
                self.remake,
                self.redistribute_on_remake,
            )
        else:
            fields.alloc_init(
                self.mf_name,
                self.level,
                (self.ba or warpx.boxArray(self.level)),
                (self.dm or warpx.DistributionMap(self.level)),
                self.ncomp,
                libwarpx.amr.IntVect(self.ngrow),
                self.initial_value,
                self.remake,
                self.redistribute_on_remake,
            )

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


def CustomNamedxWrapper(mf_name, level=0):
    return MultiFabWrapper(mf_name=mf_name, idir=0, level=level)


def CustomNamedyWrapper(mf_name, level=0):
    return MultiFabWrapper(mf_name=mf_name, idir=1, level=level)


def CustomNamedzWrapper(mf_name, level=0):
    return MultiFabWrapper(mf_name=mf_name, idir=2, level=level)


def ExWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_aux", idir=0, level=level)


def EyWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_aux", idir=1, level=level)


def EzWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_aux", idir=2, level=level)


def BxWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_aux", idir=0, level=level)


def ByWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_aux", idir=1, level=level)


def BzWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_aux", idir=2, level=level)


def JxWrapper(level=0):
    return MultiFabWrapper(mf_name="current_fp", idir=0, level=level)


def JyWrapper(level=0):
    return MultiFabWrapper(mf_name="current_fp", idir=1, level=level)


def JzWrapper(level=0):
    return MultiFabWrapper(mf_name="current_fp", idir=2, level=level)


def ExFPWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_fp", idir=0, level=level)


def EyFPWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_fp", idir=1, level=level)


def EzFPWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_fp", idir=2, level=level)


def BxFPWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_fp", idir=0, level=level)


def ByFPWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_fp", idir=1, level=level)


def BzFPWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_fp", idir=2, level=level)


def ExFPExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_fp_external", idir=0, level=level)


def EyFPExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_fp_external", idir=1, level=level)


def EzFPExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_fp_external", idir=2, level=level)


def BxFPExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_fp_external", idir=0, level=level)


def ByFPExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_fp_external", idir=1, level=level)


def BzFPExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_fp_external", idir=2, level=level)


def AxHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_A_fp_external", idir=0, level=level)


def AyHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_A_fp_external", idir=1, level=level)


def AzHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_A_fp_external", idir=2, level=level)


def ExHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_E_fp_external", idir=0, level=level)


def EyHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_E_fp_external", idir=1, level=level)


def EzHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_E_fp_external", idir=2, level=level)


def BxHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_B_fp_external", idir=0, level=level)


def ByHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_B_fp_external", idir=1, level=level)


def BzHybridExternalWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_B_fp_external", idir=2, level=level)


def JxFPWrapper(level=0):
    return MultiFabWrapper(mf_name="current_fp", idir=0, level=level)


def JyFPWrapper(level=0):
    return MultiFabWrapper(mf_name="current_fp", idir=1, level=level)


def JzFPWrapper(level=0):
    return MultiFabWrapper(mf_name="current_fp", idir=2, level=level)


def RhoFPWrapper(level=0):
    return MultiFabWrapper(mf_name="rho_fp", level=level)


def PhiFPWrapper(level=0):
    return MultiFabWrapper(mf_name="phi_fp", level=level)


def FFPWrapper(level=0):
    return MultiFabWrapper(mf_name="F_fp", level=level)


def GFPWrapper(level=0):
    return MultiFabWrapper(mf_name="G_fp", level=level)


def AxFPWrapper(level=0):
    return MultiFabWrapper(mf_name="vector_potential_fp_nodal", idir=0, level=level)


def AyFPWrapper(level=0):
    return MultiFabWrapper(mf_name="vector_potential_fp_nodal", idir=1, level=level)


def AzFPWrapper(level=0):
    return MultiFabWrapper(mf_name="vector_potential_fp_nodal", idir=2, level=level)


def ExCPWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_cp", idir=0, level=level)


def EyCPWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_cp", idir=1, level=level)


def EzCPWrapper(level=0):
    return MultiFabWrapper(mf_name="Efield_cp", idir=2, level=level)


def BxCPWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_cp", idir=0, level=level)


def ByCPWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_cp", idir=1, level=level)


def BzCPWrapper(level=0):
    return MultiFabWrapper(mf_name="Bfield_cp", idir=2, level=level)


def JxCPWrapper(level=0):
    return MultiFabWrapper(mf_name="current_cp", idir=0, level=level)


def JyCPWrapper(level=0):
    return MultiFabWrapper(mf_name="current_cp", idir=1, level=level)


def JzCPWrapper(level=0):
    return MultiFabWrapper(mf_name="current_cp", idir=2, level=level)


def RhoCPWrapper(level=0):
    return MultiFabWrapper(mf_name="rho_cp", level=level)


def FCPWrapper(level=0):
    return MultiFabWrapper(mf_name="F_cp", level=level)


def GCPWrapper(level=0):
    return MultiFabWrapper(mf_name="G_cp", level=level)


def EdgeLengthsxWrapper(level=0):
    return MultiFabWrapper(mf_name="edge_lengths", idir=0, level=level)


def EdgeLengthsyWrapper(level=0):
    return MultiFabWrapper(mf_name="edge_lengths", idir=1, level=level)


def EdgeLengthszWrapper(level=0):
    return MultiFabWrapper(mf_name="edge_lengths", idir=2, level=level)


def FaceAreasxWrapper(level=0):
    return MultiFabWrapper(mf_name="face_areas", idir=0, level=level)


def FaceAreasyWrapper(level=0):
    return MultiFabWrapper(mf_name="face_areas", idir=1, level=level)


def FaceAreaszWrapper(level=0):
    return MultiFabWrapper(mf_name="face_areas", idir=2, level=level)


def JxFPPlasmaWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_current_fp_plasma", idir=0, level=level)


def JyFPPlasmaWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_current_fp_plasma", idir=1, level=level)


def JzFPPlasmaWrapper(level=0):
    return MultiFabWrapper(mf_name="hybrid_current_fp_plasma", idir=2, level=level)


def ExFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_E_fp", idir=0, level=level)


def EyFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_E_fp", idir=1, level=level)


def EzFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_E_fp", idir=2, level=level)


def BxFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_B_fp", idir=0, level=level)


def ByFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_B_fp", idir=1, level=level)


def BzFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_B_fp", idir=2, level=level)


def JxFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_j_fp", idir=0, level=level)


def JyFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_j_fp", idir=1, level=level)


def JzFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_j_fp", idir=2, level=level)


def FFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_F_fp", level=level)


def GFPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_G_fp", level=level)


def ExCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_E_cp", idir=0, level=level)


def EyCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_E_cp", idir=1, level=level)


def EzCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_E_cp", idir=2, level=level)


def BxCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_B_cp", idir=0, level=level)


def ByCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_B_cp", idir=1, level=level)


def BzCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_B_cp", idir=2, level=level)


def JxCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_j_cp", idir=0, level=level)


def JyCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_j_cp", idir=1, level=level)


def JzCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_j_cp", idir=2, level=level)


def FCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_F_cp", level=level)


def GCPPMLWrapper(level=0):
    return MultiFabWrapper(mf_name="pml_G_cp", level=level)
