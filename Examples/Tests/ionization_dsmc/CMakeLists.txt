# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ionization_electron_dsmc  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_ionization_electron_dsmc  # inputs
    "analysis_ionization_dsmc_3d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000250"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ionization_ion_dsmc  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_ionization_ion_dsmc  # inputs
    OFF # "analysis_ionization_dsmc_3d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000250"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_ionization_neutral_dsmc  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_ionization_neutral_dsmc  # inputs
    "analysis_ionization_dsmc_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_charge_exchange_dsmc  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_charge_exchange_dsmc  # inputs
    "analysis_charge_exchange_dsmc_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag"  # checksum
    OFF  # dependency
)
