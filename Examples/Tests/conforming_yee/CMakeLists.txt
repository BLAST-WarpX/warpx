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
    set(crossing_outward "")
    set(crossing_inward "--inward")
    foreach(direction IN ITEMS outward inward)
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
