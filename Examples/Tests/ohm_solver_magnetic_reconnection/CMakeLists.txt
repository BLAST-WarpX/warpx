# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_magnetic_reconnection_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_magnetic_reconnection_picmi.py --test"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)
