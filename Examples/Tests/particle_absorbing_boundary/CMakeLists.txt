# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_particle_absorbing_boundary  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_particle_absorbing_boundary  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diagInst008000/"  # checksum
    OFF  # dependency
)
