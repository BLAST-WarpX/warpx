# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_circle  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_embedded_circle  # inputs
        analysis.py  # analysis
        diags/diag1000011
        OFF  # dependency
    )
endif()
