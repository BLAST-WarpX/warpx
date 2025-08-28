# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_linear_compton_bunch_laser  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_linear_compton_bunch_laser  # inputs
    analysis_bunch_laser.py  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_linear_compton_two_particles  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_linear_compton_two_particles  # inputs
    "analysis_two_particles.py"  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)
