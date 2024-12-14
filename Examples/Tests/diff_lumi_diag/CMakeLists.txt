# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_diff_lumi_diag_leptons  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_diff_lumi_diag_leptons  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000080 --rtol 1e-2"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_diff_lumi_diag_photons  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_diff_lumi_diag_photons  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000080 --rtol 1e-2"  # checksum
    OFF  # dependency
)
