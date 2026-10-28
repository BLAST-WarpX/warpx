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
        "analysis_default_restart.py --path diags/diag1000300"  # analysis
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
    "analysis_default_restart.py --path diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    test_2d_pml_x_yee  # dependency
)

foreach(size IN ITEMS 24 96)
    add_warpx_test(
        test_3d_pml_coincident_faces_${size}
        3
        1
        "inputs_base_3d_pml_coincident_faces.py --max-grid-size ${size}"
        OFF  # analysis (analytic damping profile assertions in the script)
        OFF  # checksum
        OFF  # dependency
    )
endforeach()

if(3 IN_LIST WarpX_DIMS AND WarpX_APP)
    add_warpx_test(
        test_3d_pml_conflicting_faces_rejected
        3
        1
        inputs_test_3d_pml_conflicting_faces_rejected
        OFF
        OFF
        OFF
    )
    set_tests_properties(
        test_3d_pml_conflicting_faces_rejected.run
        PROPERTIES PASS_REGULAR_EXPRESSION "incompatible direct-face damping profiles"
    )
endif()

foreach(refined IN ITEMS 0 1)
    foreach(regular IN ITEMS 0 1)
        foreach(pml IN ITEMS 0 1)
            add_warpx_test(
                test_3d_pml_independent_cleaning_${refined}_${regular}_${pml}
                3
                1
                "inputs_base_3d_pml_independent_cleaning.py --refined ${refined} --regular ${regular} --pml ${pml}"
                OFF  # analysis (analytic field assertions in the script)
                OFF  # checksum
                OFF  # dependency
            )
        endforeach()
    endforeach()
endforeach()

add_warpx_test(
    test_3d_magnetic_cleaning_wave
    3
    1
    inputs_base_3d_magnetic_cleaning_wave.py
    OFF
    OFF
    OFF
)

foreach(solver IN ITEMS yee ckc)
    foreach(refined IN ITEMS 0 1)
        foreach(regular IN ITEMS 0 1)
            foreach(pml IN ITEMS 0 1)
                add_warpx_test(
                    test_3d_pml_magnetic_cleaning_${solver}_${refined}_${regular}_${pml}
                    3
                    1
                    "inputs_base_3d_pml_magnetic_cleaning.py --solver ${solver} --refined ${refined} --regular ${regular} --pml ${pml}"
                    OFF
                    OFF
                    OFF
                )
            endforeach()
        endforeach()
        add_warpx_test(
            test_3d_pml_magnetic_cleaning_gradient_${solver}_${refined}
            3
            1
            "inputs_base_3d_pml_magnetic_cleaning.py --solver ${solver} --refined ${refined} --regular 0 --pml 1 --seed-g"
            OFF
            OFF
            OFF
        )
    endforeach()
endforeach()


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

add_warpx_test(
    test_2d_pml_theta_implicit
    2
    1
    inputs_test_2d_pml_theta_implicit
    analysis_pml_theta_implicit.py
    OFF
    OFF
)

add_warpx_test(
    test_rz_pml_yee_smr_picmi
    RZ
    1
    inputs_test_rz_pml_yee_smr_picmi.py
    analysis_pml_yee_rz_smr.py
    OFF
    OFF
)

add_warpx_test(
    test_rz_pml_yee_smr_picmi_restart
    RZ
    1
    "inputs_test_rz_pml_yee_smr_picmi.py --restart ../test_rz_pml_yee_smr_picmi/diags/chk000175"
    "analysis_pml_yee_rz_smr.py --reference ../test_rz_pml_yee_smr_picmi"
    OFF
    test_rz_pml_yee_smr_picmi
)

add_warpx_test(
    test_rz_pml_yee_smr_mode1_picmi
    RZ
    1
    "inputs_test_rz_pml_yee_smr_picmi.py --mode 1"
    analysis_pml_yee_rz_smr.py
    OFF
    OFF
)

add_warpx_test(
    test_rz_pml_yee_smr_annular_picmi
    RZ
    1
    "inputs_test_rz_pml_yee_smr_picmi.py --annular"
    analysis_pml_yee_rz_smr.py
    OFF
    OFF
)

add_warpx_test(
    test_rz_pml_yee_smr_mode1_material_picmi
    RZ
    1
    "inputs_test_rz_pml_yee_smr_picmi.py --mode 1 --material"
    analysis_pml_yee_rz_smr.py
    OFF
    OFF
)
