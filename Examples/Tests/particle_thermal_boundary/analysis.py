#!/usr/bin/env python3
#
# Copyright 2023-2024 Revathi Jambunathan
#
# This file is part of WarpX
#
# License: BSD-3-Clause-LBNL

"""
The script checks to see the growth in total field energy and total particle energy

The input file in 2D initializes uniform plasma (electrons,ions) with
thermal boundary condition. We do not expect the particle energy to increase
beyond 2% in the time that it takes all particles to cross the domain boundary
"""

import numpy as np

FE_rdiag = "./diags/reducedfiles/EF.txt"
init_Fenergy = np.loadtxt(FE_rdiag)[1, 2]
final_Fenergy = np.loadtxt(FE_rdiag)[-1, 2]
assert final_Fenergy / init_Fenergy < 40
assert final_Fenergy < 5.0e-5

PE_rdiag = "./diags/reducedfiles/EN.txt"
init_Penergy = np.loadtxt(PE_rdiag)[0, 2]
final_Penergy = np.loadtxt(PE_rdiag)[-1, 2]
assert abs(final_Penergy - init_Penergy) / init_Penergy < 0.02
