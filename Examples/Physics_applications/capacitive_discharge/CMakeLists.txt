# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_background_mcc_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_base_1d_picmi.py --test --pythonsolver"  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000050"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_dsmc_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_base_1d_picmi.py --test --dsmc"  # inputs
    "analysis_dsmc.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000050"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_background_mcc  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_background_mcc  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000050"  # checksum
    OFF  # dependency
)

# FIXME: can we make this single precision for now?
#add_warpx_test(
#    test_2d_background_mcc_dp_psp  # name
#    2  # dims
#    2  # nprocs
#    inputs_test_2d_background_mcc_dp_psp  # inputs
#    OFF  # analysis
#    "analysis_default_regression.py --path diags/diag1000050"  # checksum
#    OFF  # dependency
#)

add_warpx_test(
    test_2d_background_mcc_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_background_mcc_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000050 --rtol 5e-3"  # checksum
    OFF  # dependency
)
