# Copyright 2016-2022 Andrew Myers, David Grote, Maxence Thevenet
# Remi Lehe, Lorenzo Giacomel
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import re
import sys

from ._libwarpx import libwarpx
from .Algo import algo
from .Amr import amr
from .Amrex import amrex
from .Boundary import boundary
from .Bucket import Bucket
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


class WarpX(Bucket):
    """
    A Python wrapper for the WarpX C++ class
    """

    # Set the C++ WarpX interface (see _libwarpx.LibWarpX) as an extension to
    # Simulation objects. In the future, LibWarpX objects may actually be owned
    # by Simulation objects to permit multiple WarpX runs simultaneously.
    extension = libwarpx

    def create_argv_list(self, **kw):
        argv = []

        for k, v in kw.items():
            if v is not None:
                argv.append(f"{k} = {v}")

        argv += warpx.attrlist()
        argv += my_constants.attrlist()
        argv += amr.attrlist()
        argv += amrex.attrlist()
        argv += geometry.attrlist()
        argv += hybridpicmodel.attrlist()
        argv += external_vector_potential.attrlist()
        argv += boundary.attrlist()
        argv += algo.attrlist()
        argv += interpolation.attrlist()
        argv += psatd.attrlist()
        argv += eb2.attrlist()

        argv += particles.attrlist()
        for particle in particles_list:
            argv += particle.attrlist()

        argv += collisions.attrlist()
        for collision in collisions_list:
            argv += collision.attrlist()

        argv += lasers.attrlist()
        for laser in lasers_list:
            argv += laser.attrlist()

        diagnostics.diags_names = diagnostics._diagnostics_dict.keys()
        argv += diagnostics.attrlist()
        for diagnostic in diagnostics._diagnostics_dict.values():
            diagnostic.species = diagnostic._species_dict.keys()
            argv += diagnostic.attrlist()
            for species_diagnostic in diagnostic._species_dict.values():
                argv += species_diagnostic.attrlist()

        reduced_diagnostics.reduced_diags_names = (
            reduced_diagnostics._diagnostics_dict.keys()
        )
        argv += reduced_diagnostics.attrlist()
        for diagnostic in reduced_diagnostics._diagnostics_dict.values():
            argv += diagnostic.attrlist()

        for bucket in self._bucket_dict.values():
            argv += bucket.attrlist()

        return argv

    def get_bucket(self, bucket_name):
        try:
            return self._bucket_dict[bucket_name]
        except KeyError:
            bucket = Bucket(bucket_name)
            self._bucket_dict[bucket_name] = bucket
            return bucket

    @staticmethod
    def read_dims_from_file(filename, visited=None):
        """Parse an AMReX input file (and its includes) and return the
        geometry.dims value as a string or None if not found.
        """
        import os

        if visited is None:
            visited = set()
        filename = os.path.abspath(filename)
        if filename in visited:
            return None
        visited.add(filename)

        # included files
        file_pattern = re.compile(r'^\s*FILE\s*=\s*"?([^"\n\r]+)"?', re.I)
        # geometry.dims can be 1/2/3/RZ/RCYLINDER/RSPHERE
        dims_pattern = re.compile(
            r'^\s*geometry\.dims\s*=\s*"?\s*(1|2|3|RZ|RCYLINDER|RSPHERE)\s*"?', re.I
        )

        dims_value = None
        try:
            with open(filename) as f:
                for line in f:
                    # Check for FILE = ... recursively
                    m_file = file_pattern.search(line)
                    if m_file:
                        included_file = m_file.group(1).strip()
                        val = WarpX.read_dims_from_file(included_file, visited)
                        if val is not None:
                            dims_value = val
                        continue

                    # Check for geometry.dims = ...
                    m_dims = dims_pattern.search(line)
                    if m_dims:
                        dims_value = m_dims.group(1)

        except (FileNotFoundError, OSError):
            print(
                f"Error: Could not open file '{filename}'. "
                "Please check if it exists and is accessible."
            )

        return dims_value

    def load_inputs_file(self, filename):
        from .Geometry import geometry

        geometry.dims = WarpX.read_dims_from_file(filename)
        if geometry.dims is None:
            raise RuntimeError(
                f"Error: Could not find the geometry.dims in the input file '{filename}' or its included files."
            )

        libwarpx.initialize(sys.argv + [filename])

    def init(self, mpi_comm=None, **kw):
        # note: argv[0] needs to be an absolute path so it works with AMReX backtraces
        # https://github.com/AMReX-Codes/amrex/issues/3435
        argv = [sys.executable] + self.create_argv_list(**kw)
        libwarpx.initialize(argv, mpi_comm=mpi_comm)

    def evolve(self, nsteps=-1):
        libwarpx.warpx.evolve(nsteps)

    def finalize(self, finalize_mpi=1):
        libwarpx.finalize(finalize_mpi)

    def getProbLo(self, direction):
        return libwarpx.libwarpx_so.warpx_getProbLo(direction)

    def getProbHi(self, direction):
        return libwarpx.libwarpx_so.warpx_getProbHi(direction)

    def write_inputs(self, filename="inputs", **kw):
        argv = self.create_argv_list(**kw)

        # Sort the argv list to make it more human readable
        argv.sort()

        with open(filename, "w") as ff:
            prefix_old = ""
            for arg in argv:
                # This prints the name of the input group (prefix) as a header
                # before each group to make the input file more human readable
                prefix_new = re.split(r" |\.", arg)[0]
                if prefix_new != prefix_old:
                    if prefix_old != "":
                        ff.write("\n")
                    ff.write(f"# {prefix_new}\n")
                    prefix_old = prefix_new

                ff.write(f"{arg}\n")


warpx = WarpX("warpx", _bucket_dict={})
