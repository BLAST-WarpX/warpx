# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_circle  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_embedded_circle  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000011 --rtol 1e-2"  # checksum
        OFF  # dependency
    )
endif()
