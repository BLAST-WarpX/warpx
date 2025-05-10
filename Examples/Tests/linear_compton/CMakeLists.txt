# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_linear_compton_two_particles  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_linear_compton_two_particles  # inputs
    "analysis_two_photons.py diags/diag1000100"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)
