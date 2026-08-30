# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

if(WarpX_LIB)
    add_executable(
        test_hybrid_metric_transport
        test_hybrid_metric_transport.cpp)
    target_link_libraries(
        test_hybrid_metric_transport
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_hybrid_metric_transport)
        target_compile_features(
            test_hybrid_metric_transport PRIVATE cuda_std_20)
        set_target_properties(
            test_hybrid_metric_transport PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_hybrid_metric_transport PRIVATE cxx_std_20)
        set_target_properties(
            test_hybrid_metric_transport PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_hybrid_metric_transport)

    add_test(
        NAME test_hybrid_metric_transport
        COMMAND test_hybrid_metric_transport)
endif()
