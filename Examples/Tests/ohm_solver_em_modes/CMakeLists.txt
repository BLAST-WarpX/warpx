# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_ohm_solver_em_modes_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_picmi.py --test --dim 1 --bdir z"  # inputs
    analysis.py  # analysis
    diags/field_diag000250  # output
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_em_modes_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_picmi.py --test"  # inputs
    analysis_rz.py  # analysis
    diags/diag1000100  # output
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_picmi slow)
