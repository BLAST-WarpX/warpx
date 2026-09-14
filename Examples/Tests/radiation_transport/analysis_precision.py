#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Precision selections shared by radiation-transport analysis scripts."""

import argparse

import numpy as np


def add_precision_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the required field and particle precision command-line arguments."""
    parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
    parser.add_argument(
        "--particle-precision", choices=["SINGLE", "DOUBLE"], required=True
    )


def precision_dtypes(args: argparse.Namespace) -> tuple[type, type, type]:
    """Return field, particle, and cross-operation NumPy dtypes."""
    field_dtype = np.float32 if args.precision == "SINGLE" else np.float64
    particle_dtype = np.float32 if args.particle_precision == "SINGLE" else np.float64
    cross_dtype = (
        np.float32
        if args.precision == "SINGLE" or args.particle_precision == "SINGLE"
        else np.float64
    )
    return field_dtype, particle_dtype, cross_dtype
