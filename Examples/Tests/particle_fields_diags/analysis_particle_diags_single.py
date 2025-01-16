#!/usr/bin/env python3

# Copyright 2019-2022 Luca Fedeli, Yinjian Zhao, Hannah Klion
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the reduced particle diagnostics.
# The setup is a uniform plasma with electrons, protons and photons.
# Various particle and field quantities are written to file using the reduced diagnostics
# and compared with the corresponding quantities computed from the data in the plotfiles.

import analysis_particle_diags_impl as an

an.do_analysis(single_precision=True)
