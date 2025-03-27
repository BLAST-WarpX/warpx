# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_langmuir_multi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_langmuir_multi  # inputs
    "analysis_1d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_langmuir_multi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_multi  # inputs
    "analysis_2d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_langmuir_multi_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_multi_mr  # inputs
    "analysis_2d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_langmuir_multi_mr_anisotropic  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_multi_mr_anisotropic  # inputs
    "analysis_2d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_langmuir_multi_mr_momentum_conserving  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_multi_mr_momentum_conserving  # inputs
    "analysis_2d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_mr_psatd  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_langmuir_multi_mr_psatd  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_2d_langmuir_multi_nodal  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_multi_nodal  # inputs
    "analysis_2d.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_langmuir_multi_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_langmuir_multi_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_langmuir_multi_psatd  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_current_correction  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_langmuir_multi_psatd_current_correction  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_current_correction_nodal  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_langmuir_multi_psatd_current_correction_nodal  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_momentum_conserving  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_langmuir_multi_psatd_momentum_conserving  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_multiJ  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_langmuir_multi_psatd_multiJ  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_multiJ_nodal  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_langmuir_multi_psatd_multiJ_nodal  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_nodal  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_langmuir_multi_psatd_nodal  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_vay_deposition  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_langmuir_multi_psatd_vay_deposition  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_vay_deposition_nodal  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_langmuir_multi_psatd_vay_deposition_nodal  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_langmuir_multi_psatd_vay_deposition_particle_shape_4  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_langmuir_multi_psatd_vay_deposition_particle_shape_4  # inputs
        "analysis_2d.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_3d_langmuir_multi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_langmuir_multi  # inputs
    "analysis_3d.py diags/diag1000040"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_langmuir_multi_nodal  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_langmuir_multi_nodal  # inputs
    "analysis_3d.py diags/diag1000040"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_langmuir_multi_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_langmuir_multi_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_langmuir_multi_psatd  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_current_correction  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_langmuir_multi_psatd_current_correction  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_current_correction_nodal  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_langmuir_multi_psatd_current_correction_nodal  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_div_cleaning  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_langmuir_multi_psatd_div_cleaning  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_momentum_conserving  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_langmuir_multi_psatd_momentum_conserving  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_multiJ  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_langmuir_multi_psatd_multiJ  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_multiJ_nodal  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_langmuir_multi_psatd_multiJ_nodal  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_nodal  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_langmuir_multi_psatd_nodal  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_vay_deposition  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_langmuir_multi_psatd_vay_deposition  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_langmuir_multi_psatd_vay_deposition_nodal  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_langmuir_multi_psatd_vay_deposition_nodal  # inputs
        "analysis_3d.py diags/diag1000040"  # analysis
        "analysis_default_regression.py --path diags/diag1000040"  # checksum
        OFF  # dependency
    )
    label_warpx_test(test_3d_langmuir_multi_psatd_vay_deposition_nodal slow)
endif()

add_warpx_test(
    test_rz_langmuir_multi  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_langmuir_multi  # inputs
    "analysis_rz.py diags/diag1000080"  # analysis
    "analysis_default_regression.py --path diags/diag1000080"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_langmuir_multi_picmi  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_langmuir_multi_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_rz_langmuir_multi_psatd  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_langmuir_multi_psatd  # inputs
        "analysis_rz.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_langmuir_multi_psatd_current_correction  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_langmuir_multi_psatd_current_correction  # inputs
        "analysis_rz.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_rz_langmuir_multi_psatd_multiJ  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_langmuir_multi_psatd_multiJ  # inputs
        "analysis_rz.py diags/diag1000080"  # analysis
        "analysis_default_regression.py --path diags/diag1000080"  # checksum
        OFF  # dependency
    )
endif()
