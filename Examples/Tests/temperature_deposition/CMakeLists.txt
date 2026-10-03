# Native text-input tests exercise the temperature used by the hybrid solver.
add_warpx_test(
    test_3d_temperature_deposition
    3
    2
    inputs_test_3d_temperature_deposition
    "analysis.py diags/diag1000002"
    "OFF"
    OFF
)

add_warpx_test(
    test_rz_temperature_deposition
    RZ
    2
    inputs_test_rz_temperature_deposition
    "analysis.py diags/diag1000002"
    "OFF"
    OFF
)
