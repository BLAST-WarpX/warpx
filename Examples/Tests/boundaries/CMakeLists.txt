# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_particle_boundaries  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particle_boundaries  # inputs
    "analysis.py diags/diag1000008"  # analysis
    "analysis_default_regression.py --path diags/diag1000008"  # checksum
    OFF  # dependency
)
