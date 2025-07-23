import re

import numpy as np
from scipy.constants import epsilon_0


def check_charge_conservation(data):
    # Read the file that records which input options were used for this run.
    with open("./warpx_used_inputs", "r") as f:
        warpx_used_inputs = f.read()

    # Detect specific configuration flags via simple regex searches.
    # These flags determine whether the charge conservation check should run
    # and whether tolerances need to be relaxed.
    geometry_rz = re.search("geometry.dims = RZ", warpx_used_inputs)
    current_correction = re.search("psatd.current_correction = 1", warpx_used_inputs)
    current_deposition_vay = re.search(
        "algo.current_deposition = vay", warpx_used_inputs
    )
    current_deposition_esirkepov = re.search(
        "algo.current_deposition = esirkepov", warpx_used_inputs
    )
    maxwell_solver_psatd = re.search("algo.maxwell_solver = psatd", warpx_used_inputs)

    # Decide whether to perform the charge conservation check. We check with
    # current correction, Vay current deposition, and Esirkepov current deposition.
    # We do not check with Esirkepov deposition in RZ geometry, since that combination
    # currently produces larger numerical errors that need to be investigated further.
    # We also do not check with Esirkepov deposition combined with the PSATD solver,
    # since that combination does not conserve charge except for spectral order 2.
    check_charge_conservation = (
        current_correction
        or current_deposition_vay
        or (current_deposition_esirkepov and not (geometry_rz or maxwell_solver_psatd))
    )

    # Default tolerance for the infinity-norm of the relative error between div(E) and rho/eps0.
    # This is relaxed for certain deposition schemes that produce larger numerical error.
    tolerance = 1e-11
    if current_correction:
        tolerance = 1e-9
    elif current_deposition_vay:
        tolerance = 1e-3

    # If the conditions above indicate we should check charge conservation,
    # compute the infinity-norm of the relative error: max|divE - rho/eps0| / max|rho/eps0|.
    if check_charge_conservation:
        rho = data[("boxlib", "rho")].to_ndarray()
        divE = data[("boxlib", "divE")].to_ndarray()
        error_rel = np.amax(np.abs(divE - rho / epsilon_0)) / np.amax(
            np.abs(rho / epsilon_0)
        )
        print("Check charge conservation:")
        print("error_rel = {}".format(error_rel))
        print("tolerance = {}".format(tolerance))
        # Fail the test if the relative error exceeds the chosen tolerance.
        assert error_rel < tolerance
