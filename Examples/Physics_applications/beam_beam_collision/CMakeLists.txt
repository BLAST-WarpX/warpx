# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_beam_beam_collision  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_beam_beam_collision  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)
label_warpx_test(test_3d_beam_beam_collision slow)
