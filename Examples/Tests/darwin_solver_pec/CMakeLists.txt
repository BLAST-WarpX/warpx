# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_darwin_solver_pec_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_darwin_solver_pec_picmi.py --test"  # inputs
    "analysis_pec.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000030"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_darwin_solver_pec_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_darwin_solver_pec_picmi.py --test"  # inputs
    "analysis_pec.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000030"  # checksum
    OFF  # dependency
)
