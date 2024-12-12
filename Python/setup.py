#!/usr/bin/env python

# Copyright 2016-2020 Andrew Myers, David Grote, Maxence Thevenet
# Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


"""
setup.py file for WarpX
"""

import argparse
import os
import sys

from setuptools import setup

argparser = argparse.ArgumentParser(add_help=False)
argparser.add_argument(
    "--with-libwarpx",
    type=str,
    default=None,
    help="Install libwarpx with the given value as DIM. This option is only used by the GNU makefile build system.",
)
argparser.add_argument(
    "--with-lib-dir",
    type=str,
    default=None,
    help="Install with all libwarpx* binaries found in a directory.",
)
args, unknown = argparser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

allowed_dims = ["1d", "2d", "3d", "rz"]

# Allow to control options via environment vars.
# Work-around for https://github.com/pypa/setuptools/issues/1712
PYWARPX_LIB_DIR = os.environ.get("PYWARPX_LIB_DIR")

if args.with_libwarpx:
    # GNUmake
    if args.with_libwarpx not in allowed_dims:
        print("WARNING: '%s' is not an allowed WarpX DIM" % args.with_libwarpx)
    package_data = {"pywarpx": ["libwarpx.%s.so" % args.with_libwarpx]}
    data_files = []
elif args.with_lib_dir or PYWARPX_LIB_DIR:
    # CMake and Package Managers
    package_data = {"pywarpx": []}
    lib_dir = args.with_lib_dir if args.with_lib_dir else PYWARPX_LIB_DIR
    my_path = os.path.dirname(os.path.realpath(__file__))
    for dim in allowed_dims:
        lib_name = "libwarpx.%s.so" % dim
        lib_path = os.path.join(lib_dir, lib_name)
        link_name = os.path.join(my_path, "pywarpx", lib_name)
        if os.path.isfile(link_name):
            os.remove(link_name)
        if os.path.isfile(lib_path) and os.access(lib_path, os.R_OK):
            os.symlink(lib_path, link_name)
            package_data["pywarpx"].append(lib_name)
else:
    package_data = {}

setup(
    name="pywarpx",
    version="24.12",
    packages=["pywarpx"],
    package_dir={"pywarpx": "pywarpx"},
    description="""Wrapper of WarpX""",
    package_data=package_data,
    install_requires=["numpy", "picmistandard==0.33.0", "periodictable"],
    python_requires=">=3.8",
    zip_safe=False,
)
