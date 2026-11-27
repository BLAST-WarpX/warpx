# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
function(register_native_rz_material_tests)
    if(NOT WarpX_LIB OR NOT WarpX_MATERIAL_OPACITY_HDF5 OR
       NOT RZ IN_LIST WarpX_DIMS OR NOT WarpX_PRECISION STREQUAL DOUBLE OR
       NOT WarpX_PARTICLE_PRECISION STREQUAL DOUBLE)
        return()
    endif()
    add_executable(test_rz_registered_material test_rz_registered_material.cpp)
    target_link_libraries(test_rz_registered_material PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_registered_material)
        target_compile_features(test_rz_registered_material PRIVATE cuda_std_20)
    else()
        target_compile_features(test_rz_registered_material PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_rz_registered_material)
    set(table "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/material_opacity_runtime_fixtures/material_w_nonlinear.h5")
    foreach(geometry IN ITEMS axis annulus)
        foreach(backend IN ITEMS analytic table mpi)
            if(backend STREQUAL mpi AND NOT WarpX_MPI)
                continue()
            endif()
            set(name test_rz_registered_material_${geometry}_${backend})
            set(directory "${CMAKE_CURRENT_BINARY_DIR}/registered_material_${geometry}_${backend}")
            file(MAKE_DIRECTORY "${directory}")
            set(args "hybrid_pic_model.electron_composition_species=ions"
                "hybrid_pic_model.electron_composition_atomic_mass=183.84"
                "hybrid_pic_model.electron_composition_atomic_number=74")
            set(launcher)
            if(backend STREQUAL analytic)
                set(factor "sqrt(ne*183.84*amu/200)*(Te/10000)^0.25")
                list(APPEND args
                    "radiation_transport.absorption_coefficient=0"
                    "radiation_transport.planck_absorption_coefficient(x,y,z,t,photon_energy,ne,Te)=3.162277660168379*ne*183.84*amu/20*if(photon_energy<2.e-19,1,10)*(${factor})"
                    "radiation_transport.rosseland_transport_coefficient(x,y,z,t,photon_energy,ne,Te)=6.324555320336758*ne*183.84*amu/20*if(photon_energy<2.e-19,1,10)*(${factor})")
            else()
                list(APPEND args
                    "materials.names=tungsten" "materials.tungsten.species=ions"
                    "materials.tungsten.opacity_table_file=${table}"
                    "materials.tungsten.opacity_material_id=1001"
                    "materials.tungsten.opacity_material_key=manufactured-w-nonlinear"
                    "radiation_transport.require_material_metadata_consistency=1"
                    "test.reference_directory=${CMAKE_CURRENT_BINARY_DIR}/registered_material_${geometry}_analytic")
            endif()
            if(backend STREQUAL mpi)
                set(launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
            endif()
            add_test(NAME ${name} COMMAND ${launcher} $<TARGET_FILE:test_rz_registered_material>
                ${MPIEXEC_POSTFLAGS}
                "inputs_rz_registered_material_${geometry}"
                "amrex.abort_on_unused_inputs=1" "amrex.throw_exception=1" ${args})
            set_tests_properties(${name} PROPERTIES
                WORKING_DIRECTORY "${directory}"
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(NOT backend STREQUAL analytic)
                set_tests_properties(${name} PROPERTIES
                    FIXTURES_REQUIRED material_opacity_runtime_fixtures
                    DEPENDS test_rz_registered_material_${geometry}_analytic)
            endif()
            if(backend STREQUAL mpi)
                set_tests_properties(${name} PROPERTIES PROCESSORS 2 RUN_SERIAL TRUE)
            endif()
        endforeach()
    endforeach()
endfunction()
register_native_rz_material_tests()
include(cmake/NativeRZSpinerMaterialTests.cmake)
