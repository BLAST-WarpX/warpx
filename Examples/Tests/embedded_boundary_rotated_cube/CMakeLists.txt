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
        test_2d_embedded_boundary_rotated_cube_yee  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_embedded_boundary_rotated_cube algo.maxwell_solver=yee"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000068"  # checksum
        test_2d_embedded_boundary_rotated_cube  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_rotated_cube_yee_hr  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_embedded_boundary_rotated_cube algo.maxwell_solver=yee 'amr.n_cell=64 64'"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000136"  # checksum
        test_2d_embedded_boundary_rotated_cube_yee  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_rotated_cube_hr  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_embedded_boundary_rotated_cube 'amr.n_cell=64 64'"  # inputs
        "analysis_convergence_2d.py --yee-lo ../test_2d_embedded_boundary_rotated_cube_yee/diags/diag1000068 --yee-hi ../test_2d_embedded_boundary_rotated_cube_yee_hr/diags/diag1000136 --ect-lo ../test_2d_embedded_boundary_rotated_cube/diags/diag1000068 --ect-hi diags/diag1000136"  # analysis
        "analysis_default_regression.py --path diags/diag1000136"  # checksum
        test_2d_embedded_boundary_rotated_cube_yee_hr  # dependency
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
