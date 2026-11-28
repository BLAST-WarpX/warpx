# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_projection_div_cleaner_applied_field_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_projection_div_cleaner_applied_field_picmi.py  # inputs
    OFF  # analysis (efficacy assertions are embedded in the input script)
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_projection_div_cleaner_applied_field_zero_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_projection_div_cleaner_applied_field_zero_picmi.py  # inputs
    OFF  # analysis (regression assertions are embedded in the input script)
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_projection_div_cleaner_callback_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_projection_div_cleaner_callback_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_projection_div_cleaner_pec_boundary  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_projection_div_cleaner_pec_boundary  # inputs
    "analysis_pec_pmc_boundary.py pec"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_projection_div_cleaner_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_projection_div_cleaner_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_projection_div_cleaner_pmc_boundary  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_projection_div_cleaner_pmc_boundary  # inputs
    "analysis_pec_pmc_boundary.py pmc"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_projection_div_cleaner  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_projection_div_cleaner  # inputs
    "analysis.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_projection_div_cleaner_initial_analytical_field_picmi  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_projection_div_cleaner_initial_analytical_field_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000000 --rtol 5e-3"  # checksum
    OFF  # dependency
)
