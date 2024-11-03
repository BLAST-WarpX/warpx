# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_rotated_cube  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_embedded_boundary_rotated_cube  # inputs
        analysis_fields_2d.py  # analysis
        diags/diag1000068  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_rotated_cube  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_rotated_cube  # inputs
        analysis_fields_3d.py  # analysis
        diags/diag1000111  # output
        OFF  # dependency
    )
endif()
