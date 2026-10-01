# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import sys

import pywarpx


def test_inputs_file_is_the_first_argument():
    """AMReX accepts an inputs file as argv[1] only, and skips its parser
    entirely when argv[1] starts with "-". A Jupyter kernel runs as
    ``ipykernel_launcher.py -f kernel-1234.json``, so passing its sys.argv on
    would drop the inputs file and with it every parameter WarpX reads.
    """
    argv = pywarpx.warpx.amrex_argv(
        "inputs_test", ["ipykernel_launcher.py", "-f", "kernel-1234.json"]
    )

    assert argv[1] == "inputs_test"
    assert "-f" not in argv


def test_name_value_overrides_are_passed_on():
    """``python run.py max_step=10`` overrides parameters of the inputs file,
    while the interpreter's own flags stay out of AMReX's parser.
    """
    argv = pywarpx.warpx.amrex_argv(
        "inputs_test", ["run.py", "max_step=10", "-v", "--color=always"]
    )

    assert argv == [sys.executable, "inputs_test", "max_step=10"]
