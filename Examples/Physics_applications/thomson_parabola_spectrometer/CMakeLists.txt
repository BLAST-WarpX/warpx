# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_thomson_parabola_spectrometer  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_thomson_parabola_spectrometer  # inputs
    analysis_default_openpmd_regression.py  # analysis
    diags/diag1  # output
    OFF  # dependency
)
