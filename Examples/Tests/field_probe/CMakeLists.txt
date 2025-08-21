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
    test_3d_field_probe_line_extend_outside
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_extend_outside  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_field_probe_line_no_window
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_no_window  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt"  # analysis
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
    test_3d_field_probe_line_window
    3  # dims
    2  # nprocs
    inputs_test_3d_field_probe_line_window  # inputs
    "analysis_field_probe_line.py --path diags/reducedfiles/FP.txt --moving_window --fp_moving_window"  # analysis
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
