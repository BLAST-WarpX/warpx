# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_2d_averaged_galilean_psatd  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_averaged_galilean_psatd  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_averaged_galilean_psatd_hybrid  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_averaged_galilean_psatd_hybrid  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_comoving_psatd_hybrid  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_comoving_psatd_hybrid  # inputs
        analysis_default_regression.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_galilean_psatd  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd_current_correction  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_galilean_psatd_current_correction  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd_current_correction_psb  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_galilean_psatd_current_correction_psb  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd_hybrid  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_galilean_psatd_hybrid  # inputs
        analysis_default_regression.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_averaged_galilean_psatd  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_averaged_galilean_psatd  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000160  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_averaged_galilean_psatd_hybrid  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_averaged_galilean_psatd_hybrid  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000160  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_galilean_psatd  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_galilean_psatd  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000300  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_galilean_psatd_current_correction  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_galilean_psatd_current_correction  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000300  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_galilean_psatd_current_correction_psb  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_galilean_psatd_current_correction_psb  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000300  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_uniform_plasma_multiJ  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_uniform_plasma_multiJ  # inputs
        analysis_multiJ.py  # analysis
        diags/diag1000300  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_galilean_psatd  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_galilean_psatd  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_galilean_psatd_current_correction  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_galilean_psatd_current_correction  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_galilean_psatd_current_correction_psb  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_galilean_psatd_current_correction_psb  # inputs
        analysis_galilean.py  # analysis
        diags/diag1000400  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_multiJ_psatd  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_multiJ_psatd  # inputs
        analysis_default_regression.py  # analysis
        diags/diag1000025  # output
        OFF  # dependency
    )
    label_warpx_test(test_rz_multiJ_psatd slow)
endif()
