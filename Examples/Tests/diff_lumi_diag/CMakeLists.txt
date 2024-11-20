# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_diff_lumi_diag_leptons  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_diff_lumi_diag_leptons  # inputs
    analysis.py  # analysis
    diags/diag1000080  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_diff_lumi_diag_photons  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_diff_lumi_diag_photons  # inputs
    analysis.py  # analysis
    diags/diag1000080  # output
    OFF  # dependency
)
