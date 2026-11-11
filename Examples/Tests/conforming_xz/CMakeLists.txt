if(2 IN_LIST WarpX_DIMS AND WarpX_PYTHON)
    foreach(polarization IN ITEMS te tm)
        foreach(mode IN ITEMS plane oblique)
            set(flag "")
            if(mode STREQUAL "oblique")
                set(flag "--oblique")
            endif()
            set(reference test_2d_xz_slab_uniform_${polarization}_${mode}_picmi)
            add_warpx_test(
                test_2d_xz_slab_uniform_${polarization}_${mode}_picmi
                2
                1
                "inputs_test_2d_conforming_xz_picmi.py --ratio 1 1 --polarization ${polarization} ${flag}"
                OFF
                OFF
                OFF
            )
            foreach(layout IN ITEMS 1x2 2x1 2x2)
                string(REPLACE "x" " " ratio "${layout}")
                set(checkpoint "")
                if(polarization STREQUAL "te" AND mode STREQUAL "oblique" AND layout STREQUAL "2x2")
                    set(checkpoint "--checkpoint")
                endif()
                add_warpx_test(
                    test_2d_xz_slab_${layout}_${polarization}_${mode}_picmi
                    2
                    1
                    "inputs_test_2d_conforming_xz_picmi.py --ratio ${ratio} --polarization ${polarization} ${flag} ${checkpoint} --reference ../${reference}/pulse.npz"
                    OFF
                    OFF
                    ${reference}
                )
            endforeach()
        endforeach()
    endforeach()
    add_warpx_test(
        test_2d_xz_slab_filtered_charge_picmi
        2
        1
        inputs_test_2d_conforming_xz_filter_picmi.py
        OFF
        OFF
        OFF
    )
    add_warpx_test(
        test_2d_xz_slab_filter_collar_picmi
        2
        1
        "inputs_test_2d_conforming_xz_filter_picmi.py --near-interface"
        OFF
        OFF
        OFF
    )
    set_tests_properties(test_2d_xz_slab_filter_collar_picmi.run PROPERTIES
        PASS_REGULAR_EXPRESSION "A particle entered the unsupported physical-interface")
    set(filtered test_2d_xz_slab_filtered_charge_picmi)
    set(filtered_checkpoint "../${filtered}/diags/chk000002")
    add_warpx_test(
        test_2d_xz_slab_filtered_picmi_restart
        2
        1
        "inputs_test_2d_conforming_xz_filter_picmi.py --restart ${filtered_checkpoint}"
        OFF
        OFF
        ${filtered}
    )
    add_warpx_test(
        test_2d_xz_slab_changed_filter_picmi_restart
        2
        1
        "inputs_test_2d_conforming_xz_filter_picmi.py --restart ${filtered_checkpoint} --change-filter"
        OFF
        OFF
        ${filtered}
    )
    set_tests_properties(test_2d_xz_slab_changed_filter_picmi_restart.run PROPERTIES
        PASS_REGULAR_EXPRESSION "The checkpoint and current input must use the same")
    set(parent test_2d_xz_slab_2x2_te_oblique_picmi)
    set(uniform test_2d_xz_slab_uniform_te_oblique_picmi)
    set(pulse "--ratio 2 2 --polarization te --oblique")
    add_warpx_test(
        test_2d_xz_slab_split_picmi_restart
        2
        1
        "inputs_test_2d_conforming_xz_picmi.py ${pulse} --restart ../${parent}/diags/chk000075 --split-reference ../${parent}/slab_endpoint.npz --reference ../${uniform}/pulse.npz"
        OFF
        OFF
        ${parent}
    )
    add_warpx_test(
        test_2d_xz_slab_wrong_grid_picmi_restart
        2
        1
        "inputs_test_2d_conforming_xz_picmi.py ${pulse} --restart ../${parent}/diags/chk000075 --ordinary"
        OFF
        OFF
        ${parent}
    )
    set_tests_properties(test_2d_xz_slab_wrong_grid_picmi_restart.run PROPERTIES
        PASS_REGULAR_EXPRESSION "The checkpoint and current input must use the same")
    set(ordinary test_2d_xz_slab_ordinary_picmi)
    add_warpx_test(
        test_2d_xz_slab_ordinary_picmi
        2
        1
        "inputs_test_2d_conforming_xz_picmi.py ${pulse} --ordinary --checkpoint"
        OFF
        OFF
        OFF
    )
    add_warpx_test(
        test_2d_xz_slab_wrong_method_picmi_restart
        2
        1
        "inputs_test_2d_conforming_xz_picmi.py ${pulse} --restart ../${ordinary}/diags/chk000075"
        OFF
        OFF
        ${ordinary}
    )
    set_tests_properties(test_2d_xz_slab_wrong_method_picmi_restart.run PROPERTIES
        PASS_REGULAR_EXPRESSION "The checkpoint and current input must use the same")
    add_warpx_test(
        test_2d_xz_slab_particle_guard_picmi
        2
        1
        "inputs_test_2d_conforming_xz_picmi.py ${pulse} --particle-at-interface"
        OFF
        OFF
        OFF
    )
    set_tests_properties(test_2d_xz_slab_particle_guard_picmi.run PROPERTIES
        PASS_REGULAR_EXPRESSION "A particle entered the unsupported physical-interface")
endif()
