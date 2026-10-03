# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Roelof Groenewald (Realta Fusion)
#
# License: BSD-3-Clause-LBNL

from .Bucket import Bucket

# Options passed on to HYPRE by AMReX, e.g. when HYPRE is used as the bottom
# solver of the MLMG solvers (requires compiling with -DWarpX_HYPRE=ON)
hypre = Bucket("hypre")
