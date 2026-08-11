# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ohm_solver_eb_update_flags_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_eb_update_flags_picmi.py"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
