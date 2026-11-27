if(WarpX_PRECISION STREQUAL "DOUBLE" AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    set(limited_m1_runtime_procs 1)
    if(WarpX_MPI)
        set(limited_m1_runtime_procs 2)
    endif()
    add_warpx_test(
        test_3d_radiation_limited_m1_mixed
        3
        ${limited_m1_runtime_procs}
        inputs_base_3d_limited_m1_mixed_probe
        "analysis_limited_m1_mixed_probe.py ."
        OFF
        OFF
    )
    # Preserve checkpoint-surviving identities; newly born photons are compared
    # by their complete physical records, not restart-dependent allocation IDs.
    add_warpx_test(
        test_3d_radiation_limited_m1_mixed_restart
        3
        ${limited_m1_runtime_procs}
        "inputs_base_3d_limited_m1_mixed_probe amr.restart='../test_3d_radiation_limited_m1_mixed/diags/chk000040'"
        "analysis_limited_m1_mixed_probe.py . --reference ../test_3d_radiation_limited_m1_mixed"
        OFF
        test_3d_radiation_limited_m1_mixed
    )
endif()
