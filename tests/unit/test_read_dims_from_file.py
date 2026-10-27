# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import pytest

import pywarpx


@pytest.mark.parametrize("include", ["inputs_base", '"inputs_base"'])
def test_include_with_trailing_comment(include):
    """AMReX ends an unquoted value at the next blank and skips a comment
    behind it, so ``FILE = inputs_base  # shared setup`` includes
    ``inputs_base``. The geometry.dims in there selects the WarpX module that
    ``load_inputs_file`` loads.
    """
    with open("inputs_base", "w") as f:
        f.write("geometry.dims = RZ\n")
    with open("inputs_test", "w") as f:
        f.write(f"FILE = {include}  # shared setup\n")

    assert pywarpx.warpx.read_dims_from_file("inputs_test") == "RZ"
