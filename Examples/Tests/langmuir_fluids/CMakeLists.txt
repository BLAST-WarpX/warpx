# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_langmuir_fluid  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_langmuir_fluid  # inputs
    analysis_1d.py  # analysis
    diags/diag1000080  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_langmuir_fluid  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_fluid  # inputs
    analysis_2d.py  # analysis
    diags/diag1000080  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_langmuir_fluid  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_langmuir_fluid  # inputs
    analysis_3d.py  # analysis
    diags/diag1000040  # output
    OFF  # dependency
)

add_warpx_test(
    test_rz_langmuir_fluid  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_langmuir_fluid  # inputs
    analysis_rz.py  # analysis
    diags/diag1000080  # output
    OFF  # dependency
)
