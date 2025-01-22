# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_rotated_cube  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_embedded_boundary_rotated_cube  # inputs
        "analysis_fields_2d.py diags/diag1000068"  # analysis
        "analysis_default_regression.py --path diags/diag1000068"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_rotated_cube  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_rotated_cube  # inputs
        "analysis_fields_3d.py diags/diag1000111"  # analysis
        "analysis_default_regression.py --path diags/diag1000111"  # checksum
        OFF  # dependency
    )
endif()
