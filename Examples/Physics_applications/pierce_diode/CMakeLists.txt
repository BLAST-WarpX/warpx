# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_pierce_diode  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_pierce_diode  # inputs
    "analysis_pierce_diode.py diags/diag1/"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)
