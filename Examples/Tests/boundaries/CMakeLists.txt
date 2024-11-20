# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_particle_boundaries  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particle_boundaries  # inputs
    analysis.py  # analysis
    diags/diag1000008  # output
    OFF  # dependency
)
