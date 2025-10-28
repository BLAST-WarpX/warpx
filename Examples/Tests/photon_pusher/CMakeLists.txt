# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_photon_pusher  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_photon_pusher  # inputs
    "analysis.py diags/diag1000050"  # analysis
    "analysis_default_regression.py --path diags/diag1000050"  # checksum
    OFF  # dependency
)
