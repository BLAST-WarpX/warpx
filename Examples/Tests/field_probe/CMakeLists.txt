# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_field_probe  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_field_probe  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000544"  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_3d_field_probe_line_enters_moving_window
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_enters_moving_window  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --max_step 10 --enters_domain"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_extend_outside
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_extend_outside  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --expected_count 5"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_moving_window_bounds
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_moving_window_bounds  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --intervals 1 --max_step 20 --window_start 5 --window_end 15"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_no_window
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_no_window  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --expected_count 24"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_partial_window
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_partial_window  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --moving_window"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_partial_window_integrate
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_partial_window_integrate  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --moving_window --integrate"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_periodic_outside
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_periodic_outside  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --max_step 0 --expect_empty"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_window
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_window  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --moving_window --fp_moving_window --expected_count 24"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_plane_sparse_ids
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_plane_sparse_ids  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --intervals 1 --max_step 0 --expected_count 9 --positions_only"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_window_cross_boundary
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_window_cross_boundary  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
