# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_cube  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_embedded_boundary_cube  # inputs
        "analysis_fields_2d.py diags/diag1000114"  # analysis
        "analysis_default_regression.py --path diags/diag1000114"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_cube  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_cube  # inputs
        "analysis_fields.py diags/diag1000208"  # analysis
        "analysis_default_regression.py --path diags/diag1000208"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_cube_macroscopic  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_cube_macroscopic  # inputs
        "analysis_fields.py diags/diag1000208"  # analysis
        "analysis_default_regression.py --path diags/diag1000208"  # checksum
        OFF  # dependency
    )
endif()
