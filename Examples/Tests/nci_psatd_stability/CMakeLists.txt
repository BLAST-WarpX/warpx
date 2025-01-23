# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_2d_averaged_galilean_psatd  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_averaged_galilean_psatd  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_averaged_galilean_psatd_hybrid  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_averaged_galilean_psatd_hybrid  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_comoving_psatd_hybrid  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_comoving_psatd_hybrid  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000400"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_galilean_psatd  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd_current_correction  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_galilean_psatd_current_correction  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd_current_correction_psb  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_galilean_psatd_current_correction_psb  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_galilean_psatd_hybrid  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_galilean_psatd_hybrid  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000400"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_averaged_galilean_psatd  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_averaged_galilean_psatd  # inputs
        "analysis_galilean.py diags/diag1000160"  # analysis
        "analysis_default_regression.py --path diags/diag1000160 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_averaged_galilean_psatd_hybrid  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_averaged_galilean_psatd_hybrid  # inputs
        "analysis_galilean.py diags/diag1000160"  # analysis
        "analysis_default_regression.py --path diags/diag1000160 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_galilean_psatd  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_galilean_psatd  # inputs
        "analysis_galilean.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_galilean_psatd_current_correction  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_galilean_psatd_current_correction  # inputs
        "analysis_galilean.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_galilean_psatd_current_correction_psb  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_galilean_psatd_current_correction_psb  # inputs
        "analysis_galilean.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_uniform_plasma_multiJ  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_uniform_plasma_multiJ  # inputs
        "analysis_multiJ.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_galilean_psatd  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_galilean_psatd  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_galilean_psatd_current_correction  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_galilean_psatd_current_correction  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_galilean_psatd_current_correction_psb  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_galilean_psatd_current_correction_psb  # inputs
        "analysis_galilean.py diags/diag1000400"  # analysis
        "analysis_default_regression.py --path diags/diag1000400 --rtol 1e-8"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_multiJ_psatd  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_multiJ_psatd  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000025"  # checksum
        OFF  # dependency
    )
    label_warpx_test(test_rz_multiJ_psatd slow)
endif()
