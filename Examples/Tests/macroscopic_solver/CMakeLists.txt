# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_macroscopic_solver  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_macroscopic_solver  # inputs
    "analysis_fields.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_macroscopic_solver_picmi  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_macroscopic_solver_picmi.py  # inputs
    "analysis_fields.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_macroscopic_solver  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_macroscopic_solver  # inputs
    "analysis_fields_2d.py diags/diag1000100"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

foreach(mode IN ITEMS vacuum conductor refined)
    add_warpx_test(
        test_3d_macroscopic_cleaning_${mode}
        3
        1
        "inputs_base_3d_macroscopic_cleaning.py --mode ${mode}"
        OFF
        OFF
        OFF
    )
endforeach()

if(3 IN_LIST WarpX_DIMS AND WarpX_PYTHON)
    add_warpx_test(
        test_3d_macroscopic_cleaning_material_rejected
        3
        1
        "inputs_base_3d_macroscopic_cleaning.py --mode reject"
        OFF
        OFF
        OFF
    )
    set_tests_properties(
        test_3d_macroscopic_cleaning_material_rejected.run
        PROPERTIES PASS_REGULAR_EXPRESSION "macroscopic.dive_cleaning_vacuum_patches_only=1"
    )
endif()

foreach(medium IN ITEMS vacuum backwardeuler laxwendroff)
    foreach(levels IN ITEMS 1 2)
        add_warpx_test(
            test_3d_after_e_push_callback_${medium}_${levels}levels
            3
            1
            "inputs_base_3d_after_e_push_callback.py --medium ${medium} --levels ${levels}"
            OFF
            OFF
            OFF
        )
    endforeach()
endforeach()
