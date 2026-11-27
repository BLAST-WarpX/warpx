# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

cmake_minimum_required(VERSION 3.24)
project(AsymptoticDiffusionInterfaceTests LANGUAGES CXX)
enable_testing()

add_executable(
    test_asymptotic_diffusion_interface
    test_asymptotic_diffusion_interface.cpp
    ../../../../Source/Radiation/AsymptoticDiffusion.cpp
    ../../../../Source/Radiation/PacketDiffusionInterface.cpp
)
target_include_directories(
    test_asymptotic_diffusion_interface
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source
)
target_compile_features(test_asymptotic_diffusion_interface PRIVATE cxx_std_20)
set_target_properties(
    test_asymptotic_diffusion_interface PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

add_executable(test_asymptotic_diffusion_work
    ../test_asymptotic_diffusion_work.cpp
    ../../../../Source/Radiation/AsymptoticDiffusion.cpp
    ../../../../Source/Radiation/PacketDiffusionInterface.cpp)
target_include_directories(test_asymptotic_diffusion_work
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source)
target_compile_features(test_asymptotic_diffusion_work PRIVATE cxx_std_20)
foreach(test_name IN ITEMS test_asymptotic_diffusion_interface test_asymptotic_diffusion_work)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endforeach()
