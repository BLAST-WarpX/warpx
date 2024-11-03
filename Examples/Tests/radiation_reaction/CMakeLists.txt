# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_radiation_reaction  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_radiation_reaction  # inputs
    analysis.py  # analysis
    diags/diag1000064  # output
    OFF  # dependency
)
