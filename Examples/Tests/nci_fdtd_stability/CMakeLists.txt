# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_nci_corrector  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_nci_corrector  # inputs
    analysis_ncicorr.py  # analysis
    diags/diag1000600  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_nci_corrector_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_nci_corrector_mr  # inputs
    analysis_ncicorr.py  # analysis
    diags/diag1000600  # output
    OFF  # dependency
)
