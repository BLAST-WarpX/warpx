# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

from ._libwarpx import libwarpx


def load_cupy():
    """
    This is a helper for
    https://docs.cupy.dev/en/stable/user_guide/basic.html
    """
    amr = libwarpx.amr
    status = None

    if amr.Config.have_gpu:
        try:
            import cupy as cp

            xp = cp
            # Note: found and will use cupy
        except ImportError:
            status = "Warning: GPU found but cupy not available! Trying managed memory in numpy..."
            import numpy as np

            xp = np
        if amr.Config.gpu_backend == "SYCL":
            status = "Warning: SYCL GPU backend not yet implemented for Python"
            import numpy as np

            xp = np

    else:
        import numpy as np

        xp = np
        # Note: found and will use numpy
    return xp, status
