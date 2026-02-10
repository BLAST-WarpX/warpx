# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_ohm_solver_em_modes_implicit_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000010 --rtol 2e-4"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test"  # inputs
    "analysis_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000010 --rtol 1e-6"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_picmi slow)
