# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_photon_pusher  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_photon_pusher  # inputs
    analysis.py  # analysis
    diags/diag1000050  # output
    OFF  # dependency
)
