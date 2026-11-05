# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_external_field_staggering_prepare  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_external_field_staggering_prepare.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_external_field_staggering_nodal  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_external_field_staggering_nodal  # inputs
    "analysis_m4.py --case nodal diags/diag1000000"  # analysis
    OFF  # checksum
    test_3d_external_field_staggering_prepare  # dependency
)

add_warpx_test(
    test_3d_external_field_staggering_yee  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_external_field_staggering_yee  # inputs
    "analysis_m4.py --case yee diags/diag1000000"  # analysis
    OFF  # checksum
    test_3d_external_field_staggering_prepare  # dependency
)
