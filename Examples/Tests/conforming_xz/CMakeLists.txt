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
    foreach(layout IN ITEMS 1x2 2x1 2x2)
        string(REPLACE "x" " " ratio "${layout}")
        if(layout STREQUAL "1x2")
            set(shape 3)
        elseif(layout STREQUAL "2x1")
            set(shape 1)
        else()
            set(shape 2)
        endif()
        foreach(deposition IN ITEMS esirkepov villasenor)
            set(smoothing "")
            if(deposition STREQUAL "villasenor")
                set(smoothing "--filter")
            endif()
            set(name test_2d_xz_slab_crossing_${layout}_${deposition}_picmi)
            set(crossing "--ratio ${ratio} --shape ${shape} --deposition ${deposition} ${smoothing}")
            add_warpx_test(
                test_2d_xz_slab_crossing_${layout}_${deposition}_picmi
                2
                1
                "inputs_test_2d_conforming_xz_crossing_picmi.py ${crossing}"
                OFF
                OFF
                OFF
            )
            if(layout STREQUAL "1x2" AND deposition STREQUAL "villasenor")
                add_warpx_test(
                    test_2d_xz_slab_crossing_picmi_restart
                    2
                    1
                    "inputs_test_2d_conforming_xz_crossing_picmi.py ${crossing} --restart ../${name}/diags/chk000005"
                    OFF
                    OFF
                    ${name}
                )
            endif()
        endforeach()
    endforeach()
    add_warpx_test(
        test_2d_xz_slab_parent_particles_picmi_restart
        2
        1
        "inputs_test_2d_conforming_xz_filter_picmi.py --fixed-shape --restart ${filtered_checkpoint}"
        OFF
        OFF
        ${filtered}
    )
    set_tests_properties(test_2d_xz_slab_parent_particles_picmi_restart.run PROPERTIES
        PASS_REGULAR_EXPRESSION "A guarded XZ checkpoint requires every kinetic")
    add_warpx_test(
        test_2d_xz_slab_crossing_native_sources_picmi
        2
        1
        "inputs_test_2d_conforming_xz_crossing_picmi.py --ratio 2 2 --shape 2 --deposition esirkepov --native-source"
        OFF
        OFF
        OFF
    )
    add_warpx_test(
        test_2d_xz_slab_crossing_periodic_corner_picmi
        2
        1
        "inputs_test_2d_conforming_xz_crossing_picmi.py --ratio 2 2 --shape 3 --deposition villasenor --filter --periodic-corner"
        OFF
        OFF
        OFF
    )
    add_warpx_test(
        test_2d_xz_slab_fixed_vacuum_picmi
        2
        1
        inputs_test_2d_xz_slab_fixed_vacuum_picmi.py
        OFF
        OFF
        OFF
    )
    set(crossed test_2d_xz_slab_crossing_1x2_villasenor_picmi)
    add_warpx_test(
        test_2d_xz_slab_crossing_to_guarded_picmi_restart
        2
        1
        "inputs_test_2d_conforming_xz_crossing_picmi.py --ratio 1 2 --shape 3 --deposition villasenor --filter --restart ../${crossed}/diags/chk000005 --guarded-restart"
        OFF
        OFF
        ${crossed}
    )
    set_tests_properties(test_2d_xz_slab_crossing_to_guarded_picmi_restart.run PROPERTIES
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
