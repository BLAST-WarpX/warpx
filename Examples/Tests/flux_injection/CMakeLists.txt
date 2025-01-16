# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_flux_injection  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_flux_injection  # inputs
    "analysis_flux_injection_3d.py diags/diag1000002"  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_flux_injection  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_flux_injection  # inputs
    "analysis_flux_injection_rz.py diags/diag1000120"  # analysis
    "analysis_default_regression.py --path diags/diag1000120"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_flux_injection_from_eb  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_flux_injection_from_eb  # inputs
    "analysis_flux_injection_from_eb.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_flux_injection_from_eb  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_flux_injection_from_eb  # inputs
    "analysis_flux_injection_from_eb.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_flux_injection_from_eb  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_flux_injection_from_eb  # inputs
    "analysis_flux_injection_from_eb.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
