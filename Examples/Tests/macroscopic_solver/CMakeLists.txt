# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_macroscopic_solver  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_macroscopic_solver  # inputs
    "analysis_fields.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_macroscopic_solver  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_macroscopic_solver  # inputs
    "analysis_fields_2d.py diags/diag1000100"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)
