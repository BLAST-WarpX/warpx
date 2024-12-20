# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ionization_dsmc  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_ionization_dsmc  # inputs
    "analysis_ionization_dsmc_3d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000250"  # checksum
    OFF  # dependency
)
