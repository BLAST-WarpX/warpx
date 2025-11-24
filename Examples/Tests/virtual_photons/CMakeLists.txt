# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_virtual_photons  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_virtual_photons  # inputs
    analysis_virtual_photons.py  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)
