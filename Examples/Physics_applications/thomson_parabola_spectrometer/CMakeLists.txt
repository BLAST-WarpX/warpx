# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_thomson_parabola_spectrometer  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_thomson_parabola_spectrometer  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)
