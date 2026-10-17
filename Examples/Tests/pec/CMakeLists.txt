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

add_warpx_test(
    test_3d_pmc_field  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_pmc_field  # inputs
    "analysis_pec.py diags/diag1000134"  # analysis
    "analysis_default_regression.py --path diags/diag1000134"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_pec_field_insulator_implicit  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_pec_field_insulator_implicit  # inputs
    "analysis_pec_insulator_implicit.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_pec_field_insulator_implicit_restart  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_pec_field_insulator_implicit_restart  # inputs
    "analysis_pec_insulator_implicit.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    test_2d_pec_field_insulator_implicit  # dependency
)

add_warpx_test(
    test_2d_pec_absorbing_crop_picmi  # name
    2  # dims
    1  # nprocs
    "inputs_test_pec_absorbing_crop_picmi.py --dim xz --shape 1"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_pec_absorbing_crop_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_pec_absorbing_crop_picmi.py --dim rz --shape 1"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_pec_absorbing_crop_shape2_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_pec_absorbing_crop_picmi.py --dim rz --shape 2"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_pec_absorbing_crop_shape3_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_pec_absorbing_crop_picmi.py --dim rz --shape 3"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
