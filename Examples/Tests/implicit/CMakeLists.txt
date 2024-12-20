# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_semi_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_semi_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_filtered  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_filtered  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_picmi.py  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_2d_theta_implicit_strang_psatd  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_theta_implicit_strang_psatd  # inputs
        "analysis_2d_psatd.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )
endif()
