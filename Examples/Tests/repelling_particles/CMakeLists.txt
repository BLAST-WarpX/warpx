# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_repelling_particles  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_repelling_particles  # inputs
    analysis.py  # analysis
    diags/diag1000200  # output
    OFF  # dependency
)
