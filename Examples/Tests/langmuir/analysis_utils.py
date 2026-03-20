import sys

import numpy as np
from scipy.constants import epsilon_0

sys.path.append("../../../Tools/Parser/")
from input_file_parser import parse_input_file


def _get_input_bool(input_dict, key, expected_value):
    """Extract a boolean from input_dict, checking if key exists and matches expected_value."""
    value = input_dict.get(key)
    return value is not None and value[0] == expected_value


def check_charge_conservation(data):
    # Detect specific configuration flags via simple regex searches.
    # These flags determine whether the charge conservation check should run
    # and whether tolerances need to be relaxed.
    input_dict = parse_input_file("./warpx_used_inputs")
    geometry_dims_rz = _get_input_bool(input_dict, "geometry.dims", "RZ")
    current_correction = _get_input_bool(input_dict, "psatd.current_correction", "1")
    current_deposition_vay = _get_input_bool(
        input_dict, "algo.current_deposition", "vay"
    )
    current_deposition_esirkepov = _get_input_bool(
        input_dict, "algo.current_deposition", "esirkepov"
    )
    maxwell_solver_psatd = _get_input_bool(input_dict, "algo.maxwell_solver", "psatd")

    # Decide whether to perform the charge conservation check. We check with
    # current correction, Vay current deposition, and Esirkepov current deposition.
    # We do not check with Esirkepov deposition in RZ geometry, since that combination
    # currently produces larger numerical errors that need to be investigated further.
    # We also do not check with Esirkepov deposition combined with the PSATD solver,
    # since that combination does not conserve charge except for spectral order 2.
    check_charge_conservation = (
        (
            current_deposition_esirkepov
            and not (geometry_dims_rz or maxwell_solver_psatd)
        )
        or current_correction
        or current_deposition_vay
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
