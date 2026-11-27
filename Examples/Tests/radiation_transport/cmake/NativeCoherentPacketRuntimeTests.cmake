if(NOT WarpX_PRECISION STREQUAL "DOUBLE" OR
   NOT WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    return()
endif()
set(coherent_runtime_procs 1)
if(WarpX_MPI)
    set(coherent_runtime_procs 2)
endif()
add_warpx_test(test_3d_radiation_coherent_packet_mixed 3 ${coherent_runtime_procs}
    inputs_base_3d_coherent_packet_mixed_probe
    "analysis_coherent_packet_mixed_probe.py ." OFF OFF)
# No photons are born: restart preserves their identities even when the MPI
# decomposition changes. Compare complete native inventories and trajectories.
add_warpx_test(test_3d_radiation_coherent_packet_mixed_restart 3 1
    "inputs_base_3d_coherent_packet_mixed_probe amr.restart='../test_3d_radiation_coherent_packet_mixed/diags/chk000040'"
    "analysis_coherent_packet_mixed_probe.py . --reference ../test_3d_radiation_coherent_packet_mixed"
    OFF test_3d_radiation_coherent_packet_mixed)
set(coherent_counterflow
    "ions.ux=-0.0003 ions.uy=-0.0004 ions.uz=-0.0008660254037844386")
set(coherent_counterflow_analysis
    "--initial-u -0.0003 -0.0004 -0.0008660254037844386 --work-sign -")
add_warpx_test(test_3d_radiation_coherent_packet_counterflow 3 ${coherent_runtime_procs}
    "inputs_base_3d_coherent_packet_mixed_probe ${coherent_counterflow}"
    "analysis_coherent_packet_mixed_probe.py . ${coherent_counterflow_analysis}" OFF OFF)
add_warpx_test(test_3d_radiation_coherent_packet_counterflow_restart 3 1
    "inputs_base_3d_coherent_packet_mixed_probe ${coherent_counterflow} amr.restart='../test_3d_radiation_coherent_packet_counterflow/diags/chk000040'"
    "analysis_coherent_packet_mixed_probe.py . ${coherent_counterflow_analysis} --reference ../test_3d_radiation_coherent_packet_counterflow"
    OFF test_3d_radiation_coherent_packet_counterflow)
if(TEST test_3d_radiation_coherent_packet_mixed.run)
    set(coherent_checkpoint_dir "${CMAKE_CURRENT_BINARY_DIR}/coherent_checkpoint_models")
    file(MAKE_DIRECTORY ${coherent_checkpoint_dir})
    add_test(NAME test_3d_coherent_packet_checkpoint_contract COMMAND
        ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_coherent_packet_checkpoint.py
        $<TARGET_FILE:app_3d>
        ${CMAKE_BINARY_DIR}/bin/test_3d_radiation_coherent_packet_mixed/diags/chk000040
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_3d_coherent_packet_mixed_probe all)
    set_tests_properties(test_3d_coherent_packet_checkpoint_contract PROPERTIES
        WORKING_DIRECTORY ${coherent_checkpoint_dir}
        ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 60
        FIXTURES_REQUIRED coherent_scattering_checkpoint)
    set_property(TEST test_3d_radiation_coherent_packet_mixed.run APPEND PROPERTY
        FIXTURES_SETUP coherent_scattering_checkpoint)
    if(TEST test_3d_radiation_coherent_packet_mixed.cleanup)
        set_property(TEST test_3d_radiation_coherent_packet_mixed.cleanup APPEND PROPERTY
            FIXTURES_CLEANUP coherent_scattering_checkpoint)
    endif()
endif()
