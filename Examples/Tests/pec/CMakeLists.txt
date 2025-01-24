# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_pec_field  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pec_field  # inputs
    "analysis_pec.py diags/diag1000125"  # analysis
    "analysis_default_regression.py --path diags/diag1000125"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_pec_field_mr  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pec_field_mr  # inputs
    "analysis_pec_mr.py diags/diag1000125"  # analysis
    "analysis_default_regression.py --path diags/diag1000125"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_pec_particle  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pec_particle  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_pec_field_insulator  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_pec_field_insulator  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
