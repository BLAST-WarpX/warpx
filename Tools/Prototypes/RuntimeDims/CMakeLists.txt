# Runtime-dimensionality prototype
#
# Builds a single executable, warpx.unified, against AMReX compiled with
# AMREX_SPACEDIM=3 only. The executable runs 1D, 2D and 3D simulations
# selected at runtime via the geometry.dims input parameter, using the
# dimension-templated WarpX kernel headers (compiled with WARPX_DIM_RUNTIME).
#
# This target deliberately does not link lib_<dim>/ablastr_<dim>: it only
# includes the dimension-converted, self-contained kernel headers, which
# proves they do not depend on per-dimension compile definitions.

add_executable(app_unified
    main.cpp
    Simulation.cpp
    ${WarpX_SOURCE_DIR}/Source/Utils/Parser/ParserUtils.cpp
    ${WarpX_SOURCE_DIR}/Source/ablastr/utils/TextMsg.cpp
)

target_compile_definitions(app_unified PRIVATE WARPX_DIM_RUNTIME)
target_compile_features(app_unified PUBLIC cxx_std_20)
set_target_properties(app_unified PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "warpx.unified"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
target_include_directories(app_unified PRIVATE ${WarpX_SOURCE_DIR}/Source)
target_link_libraries(app_unified PRIVATE WarpX::thirdparty::amrex_3d)

# Tests: run the existing Langmuir-multi inputs at runtime-selected
# dimensionalities from the one binary (cf. Examples/Tests/langmuir), check
# the fields against the analytic plasma-wave solution and, if the native
# per-dimension binaries are built too, compare against them field by field.
if(BUILD_TESTING)
    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter QUIET)
    endif()
    set(langmuir_dir ${WarpX_SOURCE_DIR}/Examples/Tests/langmuir)
    set(analysis_dir ${CMAKE_CURRENT_SOURCE_DIR}/analysis)
    # both codes run with the same inputs overrides: direct current
    # deposition (the prototype slice) and no openPMD output
    set(test_overrides
        algo.current_deposition=direct
        diagnostics.diags_names=diag1
        diag1.intervals=40
        diag1.diag_type=Full
    )
    foreach(D IN ITEMS 1 2 3)
        set(work_dir ${CMAKE_CURRENT_BINARY_DIR}/test_${D}d)
        file(MAKE_DIRECTORY ${work_dir})
        # some inputs files use ParmParse FILE includes relative to the
        # working directory
        file(COPY ${langmuir_dir}/inputs_test_${D}d_langmuir_multi DESTINATION ${work_dir})
        if(EXISTS ${langmuir_dir}/inputs_base_${D}d)
            file(COPY ${langmuir_dir}/inputs_base_${D}d DESTINATION ${work_dir})
        endif()

        add_test(
            NAME test_unified_langmuir_${D}d.run
            COMMAND
                $<TARGET_FILE:app_unified>
                inputs_test_${D}d_langmuir_multi
                ${test_overrides}
            WORKING_DIRECTORY ${work_dir}
        )
        add_test(
            NAME test_unified_langmuir_${D}d.analysis
            COMMAND
                ${Python_EXECUTABLE} ${analysis_dir}/analysis_langmuir.py
                ${work_dir}/diags/plt00040
        )
        set_tests_properties(test_unified_langmuir_${D}d.analysis PROPERTIES
            DEPENDS test_unified_langmuir_${D}d.run
        )

        # comparison against the native per-dimension binary (serial, bitwise)
        if(TARGET app_${D}d)
            set(native_dir ${CMAKE_CURRENT_BINARY_DIR}/native_${D}d)
            file(MAKE_DIRECTORY ${native_dir})
            file(COPY ${langmuir_dir}/inputs_test_${D}d_langmuir_multi DESTINATION ${native_dir})
            if(EXISTS ${langmuir_dir}/inputs_base_${D}d)
                file(COPY ${langmuir_dir}/inputs_base_${D}d DESTINATION ${native_dir})
            endif()
            add_test(
                NAME test_unified_langmuir_${D}d.native_run
                COMMAND
                    $<TARGET_FILE:app_${D}d>
                    inputs_test_${D}d_langmuir_multi
                    ${test_overrides}
                WORKING_DIRECTORY ${native_dir}
            )
            add_test(
                NAME test_unified_langmuir_${D}d.compare_native
                COMMAND
                    ${Python_EXECUTABLE} ${analysis_dir}/compare_native.py
                    ${work_dir}/diags/plt00040
                    ${native_dir}/diags/diag1000040
                    1.e-12
            )
            set_tests_properties(test_unified_langmuir_${D}d.compare_native PROPERTIES
                DEPENDS "test_unified_langmuir_${D}d.run;test_unified_langmuir_${D}d.native_run"
            )
            # serial runs must agree bitwise: force one thread
            set_tests_properties(
                test_unified_langmuir_${D}d.run
                test_unified_langmuir_${D}d.native_run
                PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
            )
        endif()
    endforeach()
endif()
