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

add_warpx_test(
    test_3d_pml_separated_faces
    3
    1
    "inputs_base_3d_pml_coincident_faces.py --max-grid-size 192 --frame"
    OFF  # analysis in the script
    OFF
    OFF
)

set(pml_option_exterior "")
set(pml_option_interior "--in-domain")
foreach(placement IN ITEMS exterior interior)
    add_warpx_test(
        test_3d_pml_physical_boundary_${placement}
        3
        1
        "inputs_base_3d_pml_physical_boundary.py ${pml_option_${placement}}"
        OFF  # independent pulse check in the script
        OFF
        OFF
    )
    add_warpx_test(
        test_3d_pml_directional_shell_${placement}
        3
        1
        "inputs_base_3d_pml_directional_shell.py ${pml_option_${placement}}"
        OFF
        OFF
        OFF
    )
    add_warpx_test(
        test_3d_pml_directional_${placement}
        3
        1
        "inputs_base_3d_pml_physical_boundary.py ${pml_option_${placement}} --directional --reference ../test_3d_pml_physical_boundary_${placement}/pml_response.npz"
        OFF
        OFF
        test_3d_pml_physical_boundary_${placement}
    )
endforeach()
if(3 IN_LIST WarpX_DIMS AND WarpX_PYTHON)
    foreach(invalid IN ITEMS length value)
        add_warpx_test(
            test_3d_pml_directional_invalid_${invalid}
            3
            1
            "inputs_base_3d_pml_physical_boundary.py --invalid ${invalid}"
            OFF
            OFF
            OFF
        )
    endforeach()
    set_tests_properties(test_3d_pml_directional_invalid_length.run
        PROPERTIES PASS_REGULAR_EXPRESSION "needs one value per mesh direction")
    set_tests_properties(test_3d_pml_directional_invalid_value.run
        PROPERTIES PASS_REGULAR_EXPRESSION "requires positive cell counts")
endif()

if(3 IN_LIST WarpX_DIMS AND WarpX_APP)
    # The refined patch spans the periodic y direction, including the stepped
    # case where the active x/z absorbers extend into the patch's bounding box.
    set(patch_box "(abs(x)<0.5)*(abs(z)<0.5)")
    set(patch_stepped "${patch_box}*((x<0)+(z<0)>0)")
    foreach(layout IN ITEMS box stepped)
        add_warpx_test(
            test_3d_coarse_pml_periodic_${layout}
            3
            1
            "inputs_base_3d_pml_coarse_resolution amr.ref_ratio_vect=2 4 2 warpx.ref_patch_function(x,y,z)=if(${patch_${layout}},1,0) warpx.pml_ncell=2 warpx.pml_delta=2 amrex.fpe_trap_zero=1 max_step=1"
            OFF
            OFF
            OFF
        )
    endforeach()
    foreach(parameter IN ITEMS pml_ncell pml_delta)
        add_warpx_test(
            test_3d_coarse_${parameter}_unresolved
            3
            1
            "inputs_base_3d_pml_coarse_resolution warpx.${parameter}=1"
            OFF
            OFF
            OFF
        )
        set_tests_properties(
            test_3d_coarse_${parameter}_unresolved.run
            PROPERTIES PASS_REGULAR_EXPRESSION "requires warpx[.]${parameter}"
        )
    endforeach()

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
