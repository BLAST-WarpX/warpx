set(_system_memory_test_dim ${WarpX_DIMS_LAST})

add_executable(WarpXSystemMemoryInfoTest
    SystemMemoryInfoTest.cpp
)
target_link_libraries(WarpXSystemMemoryInfoTest
    PRIVATE
        ablastr_${_system_memory_test_dim}
)
add_test(NAME WarpXSystemMemoryInfoTest COMMAND WarpXSystemMemoryInfoTest)

add_executable(WarpXSystemMemoryGuardTest
    SystemMemoryGuardTest.cpp
    ../SystemMemoryGuard.cpp
)
target_link_libraries(WarpXSystemMemoryGuardTest
    PRIVATE
        ablastr_${_system_memory_test_dim}
)
add_test(NAME WarpXSystemMemoryGuardTest COMMAND WarpXSystemMemoryGuardTest)

# These standalone test executables are not part of WarpX's _ALL_TARGETS list, so
# the global CUDA setup in the top-level CMakeLists does not reach them. Apply the
# same AMReX CUDA compilation properties here so their AMReX-including sources are
# compiled as CUDA (not handed to the host compiler) in a CUDA build.
if(WarpX_COMPUTE STREQUAL CUDA)
    foreach(_system_memory_test_tgt IN ITEMS WarpXSystemMemoryInfoTest WarpXSystemMemoryGuardTest)
        setup_target_for_cuda_compilation(${_system_memory_test_tgt})
        target_compile_features(${_system_memory_test_tgt} PUBLIC cuda_std_20)
        set_target_properties(${_system_memory_test_tgt} PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON
        )
    endforeach()
endif()
