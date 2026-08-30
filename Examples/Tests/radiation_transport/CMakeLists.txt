# Add tests (alphabetical order) ##############################################
#

if(WarpX_LIB)
    add_executable(
        test_radiation_energy_update
        test_radiation_energy_update.cpp)
    target_link_libraries(
        test_radiation_energy_update
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_radiation_energy_update)
        target_compile_features(
            test_radiation_energy_update PRIVATE cuda_std_20)
        set_target_properties(
            test_radiation_energy_update PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_radiation_energy_update PRIVATE cxx_std_20)
        set_target_properties(
            test_radiation_energy_update PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_radiation_energy_update)

    add_test(
        NAME test_radiation_energy_update
        COMMAND test_radiation_energy_update)

    add_executable(
        test_radiation_kinetic_energy_update
        test_radiation_kinetic_energy_update.cpp)
    target_link_libraries(
        test_radiation_kinetic_energy_update
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_radiation_kinetic_energy_update)
        target_compile_features(
            test_radiation_kinetic_energy_update PRIVATE cuda_std_20)
        set_target_properties(
            test_radiation_kinetic_energy_update PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_radiation_kinetic_energy_update PRIVATE cxx_std_20)
        set_target_properties(
            test_radiation_kinetic_energy_update PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_radiation_kinetic_energy_update)

    add_test(
        NAME test_radiation_kinetic_energy_update
        COMMAND test_radiation_kinetic_energy_update)

    add_executable(
        test_radial_face_marching
        test_radial_face_marching.cpp)
    target_link_libraries(
        test_radial_face_marching
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_radial_face_marching)
        target_compile_features(
            test_radial_face_marching PRIVATE cuda_std_20)
        set_target_properties(
            test_radial_face_marching PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_radial_face_marching PRIVATE cxx_std_20)
        set_target_properties(
            test_radial_face_marching PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_radial_face_marching)

    add_test(
        NAME test_radial_face_marching
        COMMAND test_radial_face_marching)
endif()

# CTest's PASS_REGULAR_EXPRESSION does not require a failed child process.  Use
# the shared command-contract wrapper so each unsupported exact-streaming
# capability is required to abort before accepting a timestep.
function(add_radiation_exact_streaming_failure_test name dims inputs expected)
    if(NOT ${dims} IN_LIST WarpX_DIMS OR NOT WarpX_APP)
        return()
    endif()
    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()
    separate_arguments(input_args UNIX_COMMAND "${inputs}")
    warpx_set_suffix_dims(SD ${dims})
    set(test_working_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
    file(MAKE_DIRECTORY "${test_working_dir}")
    add_test(
        NAME ${name}.run
        COMMAND
            ${Python_EXECUTABLE}
            ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
            --expect-failure
            --expect "${expected}"
            --forbid "STEP 1 ends"
            --
            $<TARGET_FILE:app_${SD}>
            ${input_args}
            "amrex.abort_on_unused_inputs = 1"
            "amrex.throw_exception = 1"
            "amrex.the_arena_init_size = 0"
            "warpx.always_warn_immediately = 1"
            "warpx.do_dynamic_scheduling = 0"
            "warpx.serialize_initial_conditions = 1"
        WORKING_DIRECTORY "${test_working_dir}")
    set_property(
        TEST ${name}.run APPEND PROPERTY
        ENVIRONMENT
            "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/"
            "OMP_NUM_THREADS=1")
endfunction()

add_warpx_test(
    test_1d_radiation_transport_beer_lambert  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_beer_lambert  # inputs
    "analysis_beer_lambert.py --geometry 1d --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_beer_lambert_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_beer_lambert_mpi  # inputs
    "analysis_beer_lambert.py --geometry 1d --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(1 IN_LIST WarpX_DIMS AND WarpX_APP)
    add_warpx_test(
        test_1d_radiation_transport_kinetic_thermal_cold_rejected  # name
        1  # dims
        1  # nprocs
        inputs_test_1d_radiation_transport_kinetic_thermal_cold_rejected  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    set_tests_properties(
        test_1d_radiation_transport_kinetic_thermal_cold_rejected.run
        PROPERTIES
            WILL_FAIL TRUE
            PASS_REGULAR_EXPRESSION
                "heating requires at least two nondegenerate particles"
    )
endif()

add_warpx_test(
    test_1d_radiation_transport_kinetic_thermal_drift  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_kinetic_thermal_drift  # inputs
    "analysis_kinetic_thermal_drift.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic energy and momentum invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic_thermal_drift_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_kinetic_thermal_drift_mpi  # inputs
    "analysis_kinetic_thermal_drift.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic energy and momentum invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_layered_interface  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_layered_interface  # inputs
    "analysis_layered_interface.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (piecewise Beer-Lambert attenuation is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_matter_exchange  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_matter_exchange  # inputs
    "analysis_matter_radiation_exchange.py --geometry 1d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_radiation_transport_beer_lambert  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_radiation_transport_beer_lambert  # inputs
    "analysis_beer_lambert.py --geometry 2d --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_radiation_transport_matter_exchange  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_radiation_transport_matter_exchange  # inputs
    "analysis_matter_radiation_exchange.py --geometry 2d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_beer_lambert  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_beer_lambert  # inputs
    "analysis_beer_lambert.py --geometry rcyl --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_electron_thermodynamics  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_electron_thermodynamics  # inputs
    "analysis_electron_thermodynamics_exchange.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (analytic EOS exchange and conservative ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_mpi  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_rz_radiation_transport_hybrid_near_floor  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_radiation_transport_hybrid_near_floor  # inputs
    "analysis_hybrid_rz_near_floor_realization.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (metric realization and residual closure are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rz_radiation_transport_hybrid_near_floor_mpi  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_radiation_transport_hybrid_near_floor_mpi  # inputs
    "analysis_hybrid_rz_near_floor_realization.py --precision ${WarpX_PRECISION} --compare-reference"  # analysis
    OFF  # checksum (metric realization and serial equivalence are checked)
    test_rz_radiation_transport_hybrid_near_floor  # dependency
)

if(RCYLINDER IN_LIST WarpX_DIMS AND WarpX_APP
   AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_rcylinder_radiation_transport_kinetic_cold_ring  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_kinetic_cold_ring  # inputs
        "analysis_kinetic_cold_ring.py"  # analysis
        OFF  # checksum (local cylindrical moments and energy are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_radiation_transport_kinetic_cold_ring_mpi  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_radiation_transport_kinetic_cold_ring_mpi  # inputs
        "analysis_kinetic_cold_ring.py"  # analysis
        OFF  # checksum (local cylindrical moments and energy are checked)
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_latent_electron_thermodynamics  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_latent_electron_thermodynamics  # inputs
    "analysis_latent_electron_thermodynamics_exchange.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (nonlinear EOS exchange and conservative ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_matter_exchange  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_matter_exchange  # inputs
    "analysis_matter_radiation_exchange.py --geometry rcyl"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_boundary  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_boundary  # inputs
    "analysis_boundary.py --mode serial --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_boundary_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_boundary_mpi  # inputs
    "analysis_boundary.py --mode mpi --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_cartesian_cavity  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_cartesian_cavity  # inputs
    "analysis_diffusion_boundary.py --mode cavity --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_reflecting  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_reflecting  # inputs
    "analysis_diffusion_boundary.py --mode reflecting --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_vacuum  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_vacuum  # inputs
    "analysis_diffusion_boundary.py --mode vacuum --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_restart  # inputs
    "analysis_boundary_restart.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_diffusion_boundary_restart_restart amr.restart='../test_1d_radiation_transport_diffusion_boundary_restart/diags/chk000006'"  # inputs
    "analysis_boundary_restart.py --compare-reference --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum
    test_1d_radiation_transport_diffusion_boundary_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_marshak  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_marshak  # inputs
    "analysis_diffusion_boundary.py --mode marshak --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_rcylinder_radiation_transport_absorption_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_absorption_momentum  # inputs
        "analysis_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rcylinder_radiation_transport_absorption_moving_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_absorption_moving_momentum  # inputs
        "analysis_moving_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_diffusion_boundary_reflecting  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_diffusion_boundary_reflecting  # inputs
    "analysis_diffusion_boundary.py --mode reflecting --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_diffusion_boundary_vacuum  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_diffusion_boundary_vacuum  # inputs
    "analysis_diffusion_boundary.py --mode vacuum --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_diffusion_boundary_marshak  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_diffusion_boundary_marshak  # inputs
    "analysis_diffusion_boundary.py --mode marshak --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_streaming_boundary  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_streaming_boundary  # inputs
    "analysis_streaming_boundary_rcylinder.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_streaming_boundary_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_streaming_boundary_mpi  # inputs
    "analysis_streaming_boundary_rcylinder.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_layered  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_layered  # inputs
    "analysis_exact_layered_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (piecewise circular-face attenuation is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_layered_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_layered_mpi  # inputs
    "analysis_exact_layered_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (piecewise circular-face attenuation is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_omp_tiling  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_omp_tiling  # inputs
    "analysis_exact_omp_tiling_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (host material scatter is checked analytically)
    OFF  # dependency
)
if(TEST test_rcylinder_radiation_transport_exact_omp_tiling.run)
    set_property(
        TEST test_rcylinder_radiation_transport_exact_omp_tiling.run PROPERTY
        ENVIRONMENT
            "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/"
            "OMP_NUM_THREADS=4")
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_exact_axis  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_axis  # inputs
    "analysis_exact_axis_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (axis traversal and the post-axis radial crossing are analytic)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_outer  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_outer  # inputs
    "analysis_exact_outer_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (outer-face attenuation and streaming loss are checked)
    OFF  # dependency
)

add_radiation_exact_streaming_failure_test(
    test_rcylinder_radiation_transport_exact_annular_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered geometry.prob_lo=0.1"
    "requires an axis-containing RCYLINDER domain")

add_radiation_exact_streaming_failure_test(
    test_rcylinder_radiation_transport_exact_momentum_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered radiation_transport.enable_momentum_coupling=1 radiation_transport.momentum_species=ions"
    "incompatible with radiation_transport.enable_momentum_coupling=1.")

add_radiation_exact_streaming_failure_test(
    test_rcylinder_radiation_transport_exact_conversion_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered radiation_transport.enable_particle_conversion=1 radiation_transport.emission_photon_energy=1.0e-19"
    "enable_particle_conversion=1.")

add_radiation_exact_streaming_failure_test(
    test_rcylinder_radiation_transport_exact_boundary_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered boundary.particle_hi=absorbing"
    "requires an Open outer radial particle boundary.")

add_radiation_exact_streaming_failure_test(
    test_rz_radiation_transport_exact_rejected
    RZ
    inputs_base_rz_radiation_transport_exact_rejected
    "not supported in RZ geometry.")

add_radiation_exact_streaming_failure_test(
    test_rsphere_radiation_transport_exact_rejected
    RSPHERE
    inputs_base_rsphere_radiation_transport_exact_rejected
    "not supported in RSPHERE geometry.")

if(WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_rcylinder_radiation_transport_diffusion_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_diffusion_momentum  # inputs
        "analysis_diffusion_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rcylinder_radiation_transport_diffusion_multigroup_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_diffusion_multigroup_momentum  # inputs
        "analysis_diffusion_multigroup_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity  # inputs
    "analysis_species_opacity.py"  # analysis
    OFF  # checksum (species support and energy invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_planck  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity_planck  # inputs
    "analysis_species_opacity_planck.py"  # analysis
    OFF  # checksum (analytic Planck emission and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_packet  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity_packet  # inputs
    "analysis_species_opacity_packet.py"  # analysis
    OFF  # checksum (analytic attenuation and energy invariant are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_conversion  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity_conversion  # inputs
    "analysis_species_opacity_conversion.py"  # analysis
    OFF  # checksum (additive Rosseland conversion and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_table_packet  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_radiation_transport_species_opacity_table_packet radiation_transport.absorption_coefficient_table_file_foam='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_absorption_foam_3d_linear.dat'"  # inputs
    "analysis_species_opacity_table_packet.py"  # analysis
    OFF  # checksum (mixed backends, additive attenuation and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_table_lte_diffusion  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_radiation_transport_species_opacity_table_lte_diffusion radiation_transport.planck_absorption_coefficient_table_file_foam='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_planck_foam_2d_linear.dat' radiation_transport.planck_absorption_coefficient_spectral_table_file_tungsten='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_planck_tungsten_3d_linear.dat' radiation_transport.rosseland_transport_coefficient_spectral_table_file_foam='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_rosseland_foam_3d_linear.dat' radiation_transport.rosseland_transport_coefficient_table_file_tungsten='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_rosseland_tungsten_2d_linear.dat'"  # inputs
    "analysis_species_opacity_table_lte_diffusion.py"  # analysis
    OFF  # checksum (mixed table backends, cylindrical FLD and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_conversion  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_conversion  # inputs
    "analysis_lte.py --coupling hybrid --diffusion --conversion --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(1 IN_LIST WarpX_DIMS AND WarpX_APP)
    add_warpx_test(
        test_1d_radiation_transport_conversion_with_momentum_rejected  # name
        1  # dims
        1  # nprocs
        inputs_test_1d_radiation_transport_conversion_with_momentum_rejected  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    set_tests_properties(
        test_1d_radiation_transport_conversion_with_momentum_rejected.run
        PROPERTIES
            WILL_FAIL TRUE
            PASS_REGULAR_EXPRESSION
                "scalar diffusion field does not retain the converted packet momentum"
    )
endif()

add_warpx_test(
    test_1d_radiation_transport_conversion_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_conversion_mpi  # inputs
    "analysis_lte.py --coupling hybrid --diffusion --conversion --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion  # inputs
    "analysis_lte.py --coupling hybrid --diffusion --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_PRECISION STREQUAL "DOUBLE" AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_1d_radiation_transport_scattering_recoil  # name
        1  # dims
        1  # nprocs
        "inputs_base_1d_radiation_transport_scattering_recoil my_constants.alpha_control=0.0"  # inputs
        "analysis_scattering_recoil.py --planck-absorption 0.0 --precision ${WarpX_PRECISION}"  # analysis
        OFF  # checksum (analytic FLD, recoil and energy ledgers are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_1d_radiation_transport_scattering_recoil_absorption_control  # name
        1  # dims
        1  # nprocs
        "inputs_base_1d_radiation_transport_scattering_recoil my_constants.alpha_control=2.0"  # inputs
        "analysis_scattering_recoil.py --planck-absorption 2.0 --precision ${WarpX_PRECISION} --reference ../test_1d_radiation_transport_scattering_recoil"  # analysis
        OFF  # checksum (analytic FLD, recoil and energy ledgers are checked)
        test_1d_radiation_transport_scattering_recoil  # dependency
    )
endif()

add_warpx_test(
    test_1d_radiation_transport_hybrid  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_hybrid  # inputs
    "analysis.py diags/diag1000001 --coupling hybrid --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_kinetic  # inputs
    "analysis.py diags/diag1000001 --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_kinetic_mpi  # inputs
    "analysis.py diags/diag1000001 --coupling kinetic --require-multibox-residual --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic_zero_opacity_control  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_kinetic_zero_opacity_control  # inputs
    "analysis_kinetic_zero_opacity.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid  # inputs
    "analysis_lte.py --coupling hybrid --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit  # inputs
    "analysis_lte_implicit.py --coupling hybrid --multigroup --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic boundary capacity and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary_mpi  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic boundary capacity and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary_control  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary_control  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_near_floor  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_hybrid_near_floor  # inputs
    "analysis_hybrid_near_floor_realization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (below-ULP EOS realization and residual are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_near_floor_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_hybrid_near_floor_mpi  # inputs
    "analysis_hybrid_near_floor_realization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --compare-reference"  # analysis
    OFF  # checksum (below-ULP EOS realization and serial equivalence are checked)
    test_1d_radiation_transport_hybrid_near_floor  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_kinetic  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_kinetic  # inputs
    "analysis_lte.py --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_kinetic_implicit  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_kinetic_implicit  # inputs
    "analysis_lte_implicit.py --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_multigroup  # inputs
    "analysis_multigroup.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_kinetic  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_multigroup_kinetic  # inputs
    "analysis_multigroup_kinetic.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_multigroup_mpi  # inputs
    "analysis_multigroup.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_opacity_table  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_multigroup_opacity_table radiation_transport.planck_absorption_coefficient_spectral_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_multigroup_3d_linear.dat' radiation_transport.rosseland_transport_coefficient_spectral_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_multigroup_3d_linear.dat'"  # inputs
    "analysis_multigroup.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_multigroup_restart  # inputs
    "analysis_restart.py diags/diag1000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_multigroup_restart_restart amr.restart='../test_1d_radiation_transport_multigroup_restart/diags/chk000001'"  # inputs
    "analysis_restart.py diags/diag1000002 --compare-reference"  # analysis
    OFF  # checksum
    test_1d_radiation_transport_multigroup_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat' radiation_transport.planck_absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_2d_log.dat' radiation_transport.rosseland_transport_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_2d_log.dat'"  # inputs
    "analysis_opacity_table.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table_clamp  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table_clamp my_constants.n0=1.0e22 radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat' radiation_transport.planck_absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_2d_log.dat' radiation_transport.rosseland_transport_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_2d_log.dat'"  # inputs
    "analysis_opacity_table.py --alpha 100.0 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table_kinetic  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table_kinetic radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat'"  # inputs
    "analysis.py diags/diag1000001 --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table_linear  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table_linear radiation_transport.opacity_table_interpolation=linear radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat' radiation_transport.planck_absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_2d_log.dat' radiation_transport.rosseland_transport_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_2d_log.dat'"  # inputs
    "analysis_opacity_table.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_MATERIAL_OPACITY_HDF5 AND 1 IN_LIST WarpX_DIMS AND WarpX_APP)
    set(registered_opacity_fixture_dir
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/material_opacity_runtime_fixtures")
    set(registered_opacity_fixture_args
        "materials.material_a.opacity_table_file='${registered_opacity_fixture_dir}/material_a.h5' materials.material_b.opacity_table_file='${registered_opacity_fixture_dir}/material_b.h5'")
    set(registered_lte_opacity_fixture_args
        "materials.material_a.opacity_table_file='${registered_opacity_fixture_dir}/material_lte_a.h5' materials.material_b.opacity_table_file='${registered_opacity_fixture_dir}/material_lte_b.h5'")

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_order_a
        1
        1
        "inputs_base_1d_radiation_transport_registered_material_opacity_order_a ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_order_a.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_order_a_mpi
        1
        2
        "inputs_base_1d_radiation_transport_registered_material_opacity_order_a ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    if(TEST test_1d_radiation_transport_registered_material_opacity_order_a_mpi.run)
        set_tests_properties(
            test_1d_radiation_transport_registered_material_opacity_order_a_mpi.run
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endif()

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_order_b
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_order_b ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --reference ../test_1d_radiation_transport_registered_material_opacity_order_a"
        OFF
        test_1d_radiation_transport_registered_material_opacity_order_a
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_order_b.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_kinetic
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_kinetic ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --kinetic"
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_kinetic.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_kinetic_mpi
        1
        2
        "inputs_test_1d_radiation_transport_registered_material_opacity_kinetic_mpi ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --kinetic"
        OFF
        OFF
    )
    if(TEST test_1d_radiation_transport_registered_material_opacity_kinetic_mpi.run)
        set_tests_properties(
            test_1d_radiation_transport_registered_material_opacity_kinetic_mpi.run
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endif()

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_legacy_pair
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_legacy_pair materials.material_a.opacity_table_file='${registered_opacity_fixture_dir}/material_a.h5' radiation_transport.material_opacity_table_file='${registered_opacity_fixture_dir}/material_a.h5'"
        OFF
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_legacy_pair.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_lte
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_lte ${registered_lte_opacity_fixture_args}"
        "analysis_registered_material_opacity_lte.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_lte.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_lte_mpi
        1
        2
        "inputs_test_1d_radiation_transport_registered_material_opacity_lte_mpi ${registered_lte_opacity_fixture_args}"
        "analysis_registered_material_opacity_lte.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    if(TEST test_1d_radiation_transport_registered_material_opacity_lte_mpi.run)
        set_tests_properties(
            test_1d_radiation_transport_registered_material_opacity_lte_mpi.run
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endif()

    foreach(failure_case IN ITEMS mixed group_mismatch key_mismatch)
        set(failure_test_name
            "test_1d_radiation_transport_registered_material_opacity_${failure_case}_rejected")
        set(failure_working_dir
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${failure_test_name}")
        file(MAKE_DIRECTORY "${failure_working_dir}")
        add_test(
            NAME "${failure_test_name}"
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:app_1d>
                -DTEST_CASE=${failure_case}
                -DFIXTURE_DIR=${registered_opacity_fixture_dir}
                -DINPUT_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_registered_material_opacity_failure.cmake
            WORKING_DIRECTORY "${failure_working_dir}")
        set_tests_properties(
            "${failure_test_name}"
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endforeach()
endif()

add_warpx_test(
    test_1d_radiation_transport_path  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_path  # inputs
    "analysis_path.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_path_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_path_mpi  # inputs
    "analysis_path.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_restart  # inputs
    "analysis_restart.py diags/diag1000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_restart_restart amr.restart='../test_1d_radiation_transport_restart/diags/chk000001'"  # inputs
    "analysis_restart.py diags/diag1000002 --compare-reference"  # analysis
    OFF  # checksum
    test_1d_radiation_transport_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_realization_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_hybrid_realization_restart  # inputs
    "analysis_hybrid_realization_restart.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (checkpointed material-realization residual is checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_realization_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_hybrid_realization_restart_restart amr.restart='../test_1d_radiation_transport_hybrid_realization_restart/diags/chk000002'"  # inputs
    "analysis_hybrid_realization_restart.py --precision ${WarpX_PRECISION} --compare-reference"  # analysis
    OFF  # checksum (restart equivalence and cumulative residual are checked)
    test_1d_radiation_transport_hybrid_realization_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_thermalization  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_thermalization  # inputs
    "analysis_thermalization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_thermalization_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_thermalization_mpi  # inputs
    "analysis_thermalization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(RZ IN_LIST WarpX_DIMS AND WarpX_APP)
    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
        OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary_mpi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary_mpi  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
        OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary_control  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary_control  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
        OFF  # checksum (zero-opacity invariance is checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
        OFF  # checksum (zero-opacity invariance is checked)
        OFF  # dependency
    )
endif()

if(WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    if(RZ IN_LIST WarpX_DIMS AND WarpX_APP)
        add_warpx_test(
            test_rz_radiation_transport_kinetic_cold_ring  # name
            RZ  # dims
            1  # nprocs
            inputs_test_rz_radiation_transport_kinetic_cold_ring  # inputs
            "analysis_kinetic_cold_ring.py"  # analysis
            OFF  # checksum (local cylindrical moments and energy are checked)
            OFF  # dependency
        )
        add_warpx_test(
            test_rz_radiation_transport_kinetic_cold_ring_mpi  # name
            RZ  # dims
            2  # nprocs
            inputs_test_rz_radiation_transport_kinetic_cold_ring_mpi  # inputs
            "analysis_kinetic_cold_ring.py"  # analysis
            OFF  # checksum (local cylindrical moments and energy are checked)
            OFF  # dependency
        )
    endif()

    add_warpx_test(
        test_rz_radiation_transport_absorption_momentum  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_radiation_transport_absorption_momentum  # inputs
        "analysis_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_3d_radiation_transport_absorption_momentum  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_radiation_transport_absorption_momentum  # inputs
        "analysis_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(RSPHERE IN_LIST WarpX_DIMS AND WarpX_APP)
    add_warpx_test(
        test_rsphere_radiation_transport_momentum_rejected  # name
        RSPHERE  # dims
        1  # nprocs
        inputs_test_rsphere_radiation_transport_momentum_rejected  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    set_tests_properties(
        test_rsphere_radiation_transport_momentum_rejected.run
        PROPERTIES
            WILL_FAIL TRUE
            PASS_REGULAR_EXPRESSION
                "component-wise radiation-momentum validation passes"
    )
endif()

add_warpx_test(
    test_rz_radiation_transport_conversion  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_radiation_transport_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_radiation_transport_multigroup_conversion  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_radiation_transport_multigroup_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_conversion  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_radiation_transport_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_multigroup_conversion  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_radiation_transport_multigroup_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
