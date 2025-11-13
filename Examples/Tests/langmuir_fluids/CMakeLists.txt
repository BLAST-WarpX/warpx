# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_langmuir_fluid  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_langmuir_fluid  # inputs
    "analysis_1d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_langmuir_fluid  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_fluid  # inputs
    "analysis_2d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_langmuir_fluid  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_langmuir_fluid  # inputs
    "analysis_3d.py diags/diag1000040"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_langmuir_fluid  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_langmuir_fluid  # inputs
    "analysis_rz.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)
