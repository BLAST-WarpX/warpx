# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_picmi.py  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000002"  # checksum
        OFF  # dependency
    )
endif()
