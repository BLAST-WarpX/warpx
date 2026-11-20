# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_pressure_tensor_heat_flux  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pressure_tensor_heat_flux  # inputs
    "analysis.py diags/diag1000000"  # analysis
    "analysis_default_regression.py --path diags/diag1000000"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_pressure_tensor_heat_flux  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_pressure_tensor_heat_flux  # inputs
    "analysis.py diags/diag1000000"  # analysis
    "analysis_default_regression.py --path diags/diag1000000"  # checksum
    OFF  # dependency
)
