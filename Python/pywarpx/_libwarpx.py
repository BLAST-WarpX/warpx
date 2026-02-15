# Copyright 2017-2022 The WarpX Community
#
# This file is part of WarpX. It defines the wrapper functions that directly
# call the underlying compiled routines through pybind11.
#
# NOTE: We will reduce the libwarpx.py level of abstraction eventually!
# Please add new functionality directly to pybind11-bound modules
# and call them via sim.extension.libwarpx_so. ... and sim.extension.Config and
# sim.extension.warpx. ... from user code.
#
# Authors: Axel Huebl, Andrew Myers, David Grote, Remi Lehe, Weiqun Zhang
#
# License: BSD-3-Clause-LBNL

import atexit
import os
import sys

from .Geometry import geometry


class LibWarpX:
    """This class manages the WarpX classes, part of the Python module from the compiled C++ code.
    It will only load the library when it is referenced, and this can only be done after
    the geometry is defined so that the version of the library that is needed can be determined.
    Once loaded, all the settings of function call interfaces are set up.
    """

    def __init__(self):
        # Track whether amrex and warpx have been initialized
        self.initialized = False
        atexit.register(self.finalize)

        # set once libwarpx_so is loaded
        self.__version__ = None

    def __getattr__(self, attribute):
        if attribute == "libwarpx_so":
            # If the 'libwarpx_so' is referenced, load it.
            # Once loaded, it gets added to the dictionary so this code won't be called again.
            self.load_library()
            return self.__dict__[attribute]
        elif attribute == "warpx":
            # A `warpx` attribute has not yet been assigned, so `initialize_warpx` has not been called.
            raise AttributeError(
                "Trying to access libwarpx.warpx before initialize_warpx has been called!"
            )
        else:
            # For any other attribute, call the built-in routine - this should always
            # return an AttributeError.
            return self.__getattribute__(attribute)

    def _get_package_root(self):
        """
        Get the path to the installation location (where libwarpx.so would be installed).
        """
        cur = os.path.abspath(__file__)
        while True:
            name = os.path.basename(cur)
            if name == "pywarpx":
                return cur
            elif not name:
                return ""
            cur = os.path.dirname(cur)

    def load_library(self):
        if "libwarpx_so" in self.__dict__:
            raise RuntimeError(
                "Invalid attempt to load the pybind11 bindings library multiple times. "
                "Note that multiple AMReX/WarpX geometries cannot be loaded yet into the same Python process. "
                "Please write separate scripts for each geometry."
            )

        # --- Use geometry to determine whether to import the 1D, 2D, 3D or RZ version.
        # --- The geometry must be setup before the lib warpx shared object can be loaded.
        try:
            _dims = str(geometry.dims)
        except AttributeError:
            raise Exception(
                "The shared object could not be loaded. The geometry must be setup before the WarpX pybind11 module can be accessed. The geometry determines which version of the shared object to load."
            )

        if _dims == "RZ":
            self.geometry_dim = "rz"
        elif _dims == "1" or _dims == "2" or _dims == "3":
            self.geometry_dim = "%dd" % int(_dims)
        else:
            raise Exception("Undefined geometry %d" % _dims)

        try:
            if self.geometry_dim == "1d":
                import amrex.space1d as amr

                self.amr = amr
                from . import warpx_pybind_1d as cxx_1d

                self.libwarpx_so = cxx_1d
                self.dim = 1
            elif self.geometry_dim == "2d":
                import amrex.space2d as amr

                self.amr = amr
                from . import warpx_pybind_2d as cxx_2d

                self.libwarpx_so = cxx_2d
                self.dim = 2
            elif self.geometry_dim == "rz":
                import amrex.space2d as amr

                self.amr = amr
                from . import warpx_pybind_rz as cxx_rz

                self.libwarpx_so = cxx_rz
                self.dim = 2
            elif self.geometry_dim == "3d":
                import amrex.space3d as amr

                self.amr = amr
                from . import warpx_pybind_3d as cxx_3d

                self.libwarpx_so = cxx_3d
                self.dim = 3

            self.Config = self.libwarpx_so.Config
        except ImportError:
            raise Exception(
                f"Dimensionality '{self.geometry_dim}' was not compiled in this Python install. Please recompile with -DWarpX_DIMS={_dims}"
            )

        self.__version__ = self.libwarpx_so.__version__

        # Extend pybind11 types in libwarpx_so (and pyAMReX) with pure Python
        from .extensions.MultiFab import register_warpx_MultiFab_extension
        from .extensions.MultiFabRegister import (
            register_warpx_MultiFabRegister_extension,
        )
        from .extensions.MultiParticleContainer import (
            register_warpx_MultiParticleContainer_extension,
        )
        from .extensions.WarpXParticleContainer import (
            register_warpx_WarpXParticleContainer_extension,
        )

        register_warpx_MultiFab_extension(self.amr)
        register_warpx_MultiFabRegister_extension(self.libwarpx_so)
        register_warpx_MultiParticleContainer_extension(self.libwarpx_so)
        register_warpx_WarpXParticleContainer_extension(self.libwarpx_so)

    def amrex_init(self, argv, mpi_comm=None):
        if mpi_comm is None:  # or MPI is None:
            self.libwarpx_so.amrex_init(argv)
        else:
            raise Exception("mpi_comm argument not yet supported")

    def initialize(self, argv=None, mpi_comm=None):
        """
        Initialize WarpX and AMReX. Must be called before doing anything else.
        """
        if argv is None:
            argv = sys.argv
        self.amrex_init(argv, mpi_comm)
        self.warpx = self.libwarpx_so.get_instance()
        self.warpx.initialize_data()
        self.libwarpx_so.execute_python_callback("afterinit")
        self.libwarpx_so.execute_python_callback("particleloader")

        self.initialized = True

    def finalize(self, finalize_mpi=1):
        """
        Call finalize for WarpX and AMReX. Registered to run at program exit.

        After this call the simulation state is fully torn down and a new
        simulation may be started in the same Python process by calling
        :meth:`initialize` again.
        """
        if not self.initialized:
            return

        # Clear Python callbacks first, while everything is still alive.
        from pywarpx import callbacks

        callbacks.clear_all()

        # Destroy the WarpX C++ singleton (frees all simulation data).
        self.libwarpx_so.finalize()

        # Drop the Python reference to the (now-destroyed) C++ object.
        del self.warpx

        # Shut down AMReX.
        self.libwarpx_so.amrex_finalize()

        self.initialized = False

        # Reset all module-level parameter state so that a fresh simulation
        # can be configured without inheriting values from the previous run.
        self._reset_global_state()

    def _reset_global_state(self):
        """Reset module-level Bucket instances and lists to their initial state.

        This is necessary because the pywarpx parameter objects are module-level
        singletons.  Without resetting them, a second simulation started in the
        same interpreter would inherit stale parameters from the first run.
        """
        from .Algo import algo
        from .Amr import amr
        from .Amrex import amrex
        from .Boundary import boundary
        from .Collisions import collisions, collisions_list
        from .Constants import my_constants
        from .Diagnostics import diagnostics, reduced_diagnostics
        from .EB2 import eb2
        from .Geometry import geometry
        from .HybridPICModel import external_vector_potential, hybridpicmodel
        from .Interpolation import interpolation
        from .Lasers import lasers, lasers_list
        from .Particles import particles, particles_list
        from .PSATD import psatd
        from .WarpX import warpx

        # Clear all Bucket parameter state.
        for bucket in [
            warpx, algo, amr, amrex, boundary, collisions,
            my_constants, diagnostics, reduced_diagnostics, eb2,
            geometry, hybridpicmodel, external_vector_potential,
            interpolation, lasers, psatd,
        ]:
            bucket.clear()

        # Restore default attributes that other code may expect to exist.
        particles.clear()
        particles.species_names = []
        particles.rigid_injected_species = []

        lasers.names = []

        # Clear accumulated species, laser, and collision lists.
        particles_list.clear()
        lasers_list.clear()
        collisions_list.clear()

        # Clear diagnostic registries.
        diagnostics._diagnostics_dict.clear()
        reduced_diagnostics._diagnostics_dict.clear()

        # Clear the dynamic bucket dictionary.
        warpx._bucket_dict.clear()


libwarpx = LibWarpX()
