# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_darwin_solver_em_modes_picmi  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_darwin_solver_em_modes_picmi.py --test --dim 1 --bdir z"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000500"  # checksum
    OFF  # dependency
)
