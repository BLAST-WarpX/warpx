# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_linear_breit_wheeler_angular  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_linear_breit_wheeler_angular  # inputs
    "analysis_angular.py"  # analysis
    "analysis_default_regression.py --path diags/diag1 --skip-fields"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_linear_breit_wheeler_energy  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_linear_breit_wheeler_energy  # inputs
    "analysis_energy.py"  # analysis
    "analysis_default_regression.py --path diags/diag1 --skip-fields"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_linear_breit_wheeler_many_photons  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_linear_breit_wheeler_many_photons  # inputs
    "analysis_many_photons.py diags/diag1000050"  # analysis
    "analysis_default_regression.py --path diags/diag1000050"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_linear_breit_wheeler_two_photons  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_linear_breit_wheeler_two_photons  # inputs
    "analysis_two_photons.py diags/diag1000100"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)
