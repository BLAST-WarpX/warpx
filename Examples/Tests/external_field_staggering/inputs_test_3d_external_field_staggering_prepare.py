#!/usr/bin/env python3
"""Prepare step: write the screw-pinch equilibrium openPMD files
(ext_fields_nodal.h5 and ext_fields_yee.h5) into the test directory."""

from generate_equilibrium_fields import main

main(".")
