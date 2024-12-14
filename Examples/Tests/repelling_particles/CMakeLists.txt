# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_repelling_particles  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_repelling_particles  # inputs
    "analysis.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)
