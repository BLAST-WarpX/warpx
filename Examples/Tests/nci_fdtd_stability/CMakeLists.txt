# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_nci_corrector  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_nci_corrector  # inputs
    "analysis_ncicorr.py diags/diag1000600"  # analysis
    "analysis_default_regression.py --path diags/diag1000600"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_nci_corrector_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_nci_corrector_mr  # inputs
    "analysis_ncicorr.py diags/diag1000600"  # analysis
    "analysis_default_regression.py --path diags/diag1000600"  # checksum
    OFF  # dependency
)
