# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_picmi.py  # inputs
        analysis.py  # analysis
        diags/diag1000002  # output
        OFF  # dependency
    )
endif()
