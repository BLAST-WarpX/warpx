# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_particle_pusher  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_particle_pusher  # inputs
    "analysis.py diags/diag1010000"  # analysis
    "analysis_default_regression.py --path diags/diag1010000"  # checksum
    OFF  # dependency
)
