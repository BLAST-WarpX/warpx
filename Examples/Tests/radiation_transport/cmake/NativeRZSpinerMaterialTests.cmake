# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
function(register_native_rz_spiner_material_tests)
    if(NOT TARGET test_rz_registered_material OR NOT WarpX_SINGULARITY_EOS)
        return()
    endif()
    set(fixture_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rz_registered_material_eos")
    set(eos_table "${fixture_dir}/electron_w.sp5")
    set(opacity_table "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/material_opacity_runtime_fixtures/material_w_nonlinear.h5")
    add_test(NAME rz_registered_material_eos_fixture COMMAND ${CMAKE_COMMAND} -E env
        "WARPX_SPINER_FIXTURE_PATH=${eos_table}"
        "WARPX_SPINER_FIXTURE_ATOMIC_MASS=183.84"
        "WARPX_SPINER_FIXTURE_ATOMIC_NUMBER=74"
        "WARPX_SPINER_FIXTURE_RHO_MIN=0.001"
        "WARPX_SPINER_FIXTURE_NUM_T=512"
        $<TARGET_FILE:test_hybrid_electron_thermodynamics_spiner>)
    set_tests_properties(rz_registered_material_eos_fixture PROPERTIES
        FIXTURES_SETUP rz_registered_material_eos ENVIRONMENT "OMP_NUM_THREADS=1")
    foreach(geometry IN ITEMS axis annulus)
        foreach(variant IN ITEMS positive negative mpi)
            if(variant STREQUAL mpi AND NOT WarpX_MPI)
                continue()
            endif()
            set(table "${eos_table}")
            if(variant STREQUAL negative)
                set(table "${fixture_dir}/electron_w_negative_reference.sp5")
            endif()
            set(name test_rz_registered_material_eos_${geometry}_${variant})
            set(directory "${CMAKE_CURRENT_BINARY_DIR}/registered_material_eos_${geometry}_${variant}")
            file(MAKE_DIRECTORY "${directory}")
            # This nonlinear synthetic EOS has much greater heat capacity than
            # ideal electrons. Drive a substantial caloric transient, not a
            # vanishing perturbation of its large background internal energy.
            set(args "my_constants.radiation_excitation=100"
                "hybrid_pic_model.electron_thermodynamics=singularity_spiner"
                "hybrid_pic_model.electron_eos_species=ions"
                "hybrid_pic_model.electron_eos_ions_table_file=${table}"
                "hybrid_pic_model.electron_eos_ions_material_id=7400"
                "materials.names=tungsten" "materials.tungsten.species=ions"
                "materials.tungsten.electron_eos_table_file=${table}"
                "materials.tungsten.electron_eos_material_id=7400"
                "materials.tungsten.opacity_table_file=${opacity_table}"
                "materials.tungsten.opacity_material_id=1001"
                "materials.tungsten.opacity_material_key=manufactured-w-nonlinear"
                "radiation_transport.require_material_metadata_consistency=1")
            set(launcher)
            if(variant STREQUAL mpi)
                set(launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
            endif()
            if(NOT variant STREQUAL positive)
                list(APPEND args "test.reference_directory=${CMAKE_CURRENT_BINARY_DIR}/registered_material_eos_${geometry}_positive")
            endif()
            add_test(NAME ${name} COMMAND ${launcher} $<TARGET_FILE:test_rz_registered_material>
                ${MPIEXEC_POSTFLAGS} "inputs_rz_registered_material_${geometry}"
                "amrex.abort_on_unused_inputs=1" "amrex.throw_exception=1" ${args})
            set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY "${directory}"
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/"
                FIXTURES_REQUIRED "rz_registered_material_eos;material_opacity_runtime_fixtures")
            if(variant STREQUAL mpi)
                set_tests_properties(${name} PROPERTIES PROCESSORS 2 RUN_SERIAL TRUE)
            endif()
            if(NOT variant STREQUAL positive)
                set_tests_properties(${name} PROPERTIES
                    DEPENDS test_rz_registered_material_eos_${geometry}_positive)
            endif()
        endforeach()
    endforeach()
endfunction()
register_native_rz_spiner_material_tests()
