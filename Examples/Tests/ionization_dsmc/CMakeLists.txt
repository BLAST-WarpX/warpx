# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ionization_dsmc  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_ionization_dsmc  # inputs
    analysis_ionization_dsmc_3d.py  # analysis
    diags/diag1/  # output
    OFF  # dependency
)
