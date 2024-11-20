# Copyright 2016-2022 Andrew Myers, David Grote, Maxence Thevenet
# Remi Lehe, Lorenzo Giacomel
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import re
import sys

from . import Particles
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
from .HybridPICModel import hybridpicmodel
from .Interpolation import interpolation
from .Lasers import lasers, lasers_list
from .Particles import particles, particles_list
from .ProjectionDivBCleaner import projectiondivbcleaner
from .PSATD import psatd


class WarpX(Bucket):
    """
    A Python wrapper for the WarpX C++ class
    """

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
        argv += boundary.attrlist()
        argv += algo.attrlist()
        argv += interpolation.attrlist()
        argv += projectiondivbcleaner.attrlist()
        argv += psatd.attrlist()
        argv += eb2.attrlist()

        # --- Search through species_names and add any predefined particle objects in the list.
        particles_list_names = [p.instancename for p in particles_list]
        for pstring in particles.species_names:
            if pstring in particles_list_names:
                # --- The species is already included in particles_list
                continue
            elif hasattr(Particles, pstring):
                # --- Add the predefined species to particles_list
                particles_list.append(getattr(Particles, pstring))
                particles_list_names.append(pstring)
            else:
                raise Exception(
                    "Species %s listed in species_names not defined" % pstring
                )

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
