# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_pml_x_ckc  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_pml_x_ckc  # inputs
    "analysis_pml_ckc.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_2d_pml_x_galilean  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_pml_x_galilean  # inputs
        "analysis_pml_psatd.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_pml_x_psatd  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_pml_x_psatd  # inputs
        "analysis_pml_psatd.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_pml_x_psatd_restart  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_pml_x_psatd_restart  # inputs
        "analysis_default_restart.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300"  # checksum
        test_2d_pml_x_psatd  # dependency
    )
endif()

add_warpx_test(
    test_2d_pml_x_yee  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_pml_x_yee  # inputs
    "analysis_pml_yee.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_pml_x_yee_restart  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_pml_x_yee_restart  # inputs
    "analysis_default_restart.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    test_2d_pml_x_yee  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_3d_pml_psatd_dive_divb_cleaning  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_pml_psatd_dive_divb_cleaning  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000100"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_pml_psatd  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_pml_psatd  # inputs
        "analysis_pml_psatd_rz.py diags/diag1000500"  # analysis
        "analysis_default_regression.py --path diags/diag1000500"  # checksum
        OFF  # dependency
    )
endif()
