if(3 IN_LIST WarpX_DIMS AND WarpX_PYTHON AND WarpX_PRECISION STREQUAL "DOUBLE")
    foreach(layout IN ITEMS box stepped)
        add_warpx_test(
            test_3d_conforming_yee_${layout}_pulse
            3
            1
            "inputs_base_3d_conforming_yee.py --layout ${layout}"
            OFF
            OFF
            OFF
        )
    endforeach()
    set(invalid_gather "gathering requires the matching fixed-shape current")
    set(invalid_current "current requires the 3D conforming Yee interface")
    set(invalid_cleaning "disabled divergence cleaning")
    foreach(option IN ITEMS gather current cleaning)
        add_warpx_test(
            test_3d_conforming_yee_invalid_${option}
            3
            1
            "inputs_base_3d_conforming_yee.py --layout stepped --invalid ${option}"
            OFF
            OFF
            OFF
        )
        set_tests_properties(test_3d_conforming_yee_invalid_${option}.run
            PROPERTIES PASS_REGULAR_EXPRESSION "${invalid_${option}}")
    endforeach()
    set(crossing_outward "")
    set(crossing_inward "--inward")
    set(crossing_frozen "--frozen")
    set(crossing_boundary "--boundary")
    foreach(direction IN ITEMS outward inward frozen boundary)
        add_warpx_test(
            test_3d_conforming_yee_crossing_${direction}
            3
            1
            "inputs_base_3d_conforming_yee_crossing.py ${crossing_${direction}}"
            OFF
            OFF
            OFF
        )
    endforeach()
    set_tests_properties(test_3d_conforming_yee_crossing_boundary.run
        PROPERTIES PASS_REGULAR_EXPRESSION "A fixed-shape particle source stencil reached")
    add_warpx_test(
        test_3d_conforming_yee_stepped_absorber
        3
        1
        "inputs_base_3d_conforming_yee.py --layout stepped --absorber"
        OFF
        OFF
        OFF
    )
    add_warpx_test(
        test_3d_conforming_yee_stepped_restart
        3
        1
        "inputs_base_3d_conforming_yee.py --layout stepped --absorber --restart ../test_3d_conforming_yee_stepped_absorber"
        OFF
        OFF
        test_3d_conforming_yee_stepped_absorber
    )
endif()
