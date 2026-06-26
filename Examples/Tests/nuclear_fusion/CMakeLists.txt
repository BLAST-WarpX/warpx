# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_proton_boron_fusion  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_proton_boron_fusion  # inputs
    "analysis_proton_boron_fusion.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_deuterium_deuterium_fusion  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_deuterium_deuterium_fusion  # inputs
    "analysis_two_product_fusion.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_deuterium_deuterium_fusion_intraspecies  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_deuterium_deuterium_fusion_intraspecies  # inputs
    "analysis_deuterium_deuterium_3d_intraspecies.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_deuterium_tritium_fusion  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_deuterium_tritium_fusion  # inputs
    "analysis_two_product_fusion.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_proton_boron_fusion  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_proton_boron_fusion  # inputs
    "analysis_proton_boron_fusion.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_deuterium_tritium_fusion  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_deuterium_tritium_fusion  # inputs
    "analysis_two_product_fusion.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
