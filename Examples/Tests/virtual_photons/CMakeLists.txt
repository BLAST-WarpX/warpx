# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_beamsize_effect  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_beamsize_effect  # inputs
    analysis_beamsize_effect.py  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_virtual_photons  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_virtual_photons  # inputs
    analysis_virtual_photons.py  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_virtual_photons_muon  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_virtual_photons_muon  # inputs
    analysis_virtual_photons.py  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_virtual_photons_ion  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_virtual_photons_ion  # inputs
    "analysis_virtual_photons.py --charge_multiple 2"  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)
