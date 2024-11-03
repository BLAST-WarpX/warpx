# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_pec_field  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pec_field  # inputs
    analysis_pec.py  # analysis
    diags/diag1000125  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_pec_field_mr  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pec_field_mr  # inputs
    analysis_pec_mr.py  # analysis
    diags/diag1000125  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_pec_particle  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pec_particle  # inputs
    analysis_default_regression.py  # analysis
    diags/diag1000020  # output
    OFF  # dependency
)
