if(3 IN_LIST WarpX_DIMS AND WarpX_APP)
    foreach(patch IN ITEMS fp cp)
        if(patch STREQUAL "fp")
            set(options "")
            set(invalid_cells "4 16 16")
        else()
            set(options "amr.max_level=1 warpx.fine_tag_lo=-0.25 -0.5 -0.5 warpx.fine_tag_hi=0.25 0.5 0.5")
            set(invalid_cells "16 16 16")
        endif()
        add_warpx_test(
            test_3d_vector_guards_${patch}_rejected
            3
            1
            "inputs_base_3d_vector_guards ${options} amr.n_cell=${invalid_cells}"
            OFF OFF OFF
        )
        set_tests_properties(
            test_3d_vector_guards_${patch}_rejected.run
            PROPERTIES PASS_REGULAR_EXPRESSION
                "Efield_${patch}.*the number of guard cells"
        )
        add_warpx_test(
            test_3d_vector_guards_${patch}_valid
            3
            1
            "inputs_base_3d_vector_guards ${options} amr.n_cell=32 32 32"
            OFF OFF OFF
        )
    endforeach()

    # Filtering enlarges current guards without changing field guards.
    add_warpx_test(
        test_3d_vector_guards_current_rejected
        3
        1
        "inputs_base_3d_vector_guards amr.n_cell=6 16 16 warpx.use_filter=1 warpx.filter_npass_each_dir=2 2 2"
        OFF OFF OFF
    )
    set_tests_properties(
        test_3d_vector_guards_current_rejected.run
        PROPERTIES PASS_REGULAR_EXPRESSION
            "current_fp.*the number of guard cells"
    )
    add_warpx_test(
        test_3d_vector_guards_no_field_solver
        3
        1
        "inputs_base_3d_vector_guards amr.n_cell=6 9 1 amr.blocking_factor=1 algo.maxwell_solver=none warpx.const_dt=1e-12 guard_probe.do_not_push=1 guard_probe.do_not_deposit=1"
        OFF OFF OFF
    )
endif()
