# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ion_beam_extraction  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_ion_beam_extraction  # inputs
    "analysis_ion_beam_extraction.py diags/diag1/"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)
