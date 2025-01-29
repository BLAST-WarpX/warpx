# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.24.0)
project(WarpX VERSION 25.01)

include(${WarpX_SOURCE_DIR}/cmake/WarpXFunctions.cmake)

# In-source tree builds are messy and can screw up the build system.
# Avoid building at least in the same dir as the root dir:
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()


# CMake policies ##############################################################
#
# Setting a cmake_policy to OLD is deprecated by definition and will raise a
# verbose warning
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
endif()

# AMReX 21.06+ supports CUDA_ARCHITECTURES with CMake 3.20+
# CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()


# C++ Standard in Superbuilds #################################################
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
set_cxx17_superbuild()


# CCache Support ##############################################################
#
# this is an optional tool that stores compiled object files; allows fast
# re-builds even with "make clean" in between. Mainly used to store AMReX
# objects
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(WarpX_CCACHE_DEFAULT ON)
else()
    set(WarpX_CCACHE_DEFAULT OFF)  # we are a subproject in a superbuild
endif()
option(WarpX_CCACHE "Enable ccache for faster rebuilds" ${WarpX_CCACHE_DEFAULT})
if(WarpX_CCACHE)
    set_ccache()
endif()


# Output Directories ##########################################################
#
# temporary build directories
set_default_build_dirs()

# install directories
set_default_install_dirs()


# Options and Variants ########################################################
#
include(CMakeDependentOption)
option(WarpX_APP           "Build the WarpX executable application"     ON)
option(WarpX_ASCENT        "Ascent in situ diagnostics"                 OFF)
option(WarpX_CATALYST      "Catalyst in situ diagnostics"               OFF)
option(WarpX_EB            "Embedded boundary support"                  ON)
option(WarpX_LIB           "Build WarpX as a library"                   OFF)
option(WarpX_MPI           "Multi-node support (message-passing)"       ON)
option(WarpX_OPENPMD       "openPMD I/O (HDF5, ADIOS)"                  ON)
option(WarpX_FFT           "FFT-based solvers"                          OFF)
option(WarpX_PYTHON        "Python bindings"                            OFF)
option(WarpX_SENSEI        "SENSEI in situ diagnostics"                 OFF)
option(WarpX_QED           "QED support (requires PICSAR)"              ON)
option(WarpX_QED_TABLE_GEN "QED table generation (requires PICSAR and Boost)"
                                                                        OFF)
option(WarpX_QED_TOOLS     "Build external tool to generate QED lookup tables (requires PICSAR and Boost)"
                                                                        OFF)

# Advanced option to automatically clean up CI test directories
option(WarpX_TEST_CLEANUP "Clean up CI test directories" OFF)
mark_as_advanced(WarpX_TEST_CLEANUP)

# Advanced option to run CI tests with FPE-trapping runtime parameters
option(WarpX_TEST_FPETRAP "Run CI tests with FPE-trapping runtime parameters" OFF)
mark_as_advanced(WarpX_TEST_FPETRAP)

# Advanced option to run CI tests with the -g compile option
option(WarpX_TEST_DEBUG "Run CI tests with the -g compile option" OFF)
mark_as_advanced(WarpX_TEST_DEBUG)
if(WarpX_TEST_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g1")
endif()

set(WarpX_DIMS_VALUES 1 2 3 RZ)
set(WarpX_DIMS 3 CACHE STRING "Simulation dimensionality <1;2;3;RZ>")
list(REMOVE_DUPLICATES WarpX_DIMS)
foreach(D IN LISTS WarpX_DIMS)
    if(NOT D IN_LIST WarpX_DIMS_VALUES)
       message(FATAL_ERROR "WarpX_DIMS=${D} is not allowed."
          " Must be one of ${WarpX_DIMS_VALUES}")
    endif()
endforeach()

# for some targets need to be triggered once, so any dim dependency will do
list(LENGTH WarpX_DIMS list_len)
math(EXPR list_last "${list_len} - 1")
list(GET WarpX_DIMS ${list_last} WarpX_DIMS_LAST)
warpx_set_suffix_dims(WarpX_DIMS_LAST ${WarpX_DIMS_LAST})

set(WarpX_PRECISION_VALUES SINGLE DOUBLE)
set(WarpX_PRECISION DOUBLE CACHE STRING "Floating point precision (SINGLE/DOUBLE)")
set_property(CACHE WarpX_PRECISION PROPERTY STRINGS ${WarpX_PRECISION_VALUES})
if(NOT WarpX_PRECISION IN_LIST WarpX_PRECISION_VALUES)
    message(FATAL_ERROR "WarpX_PRECISION (${WarpX_PRECISION}) must be one of ${WarpX_PRECISION_VALUES}")
endif()

set(WarpX_PARTICLE_PRECISION_VALUES SINGLE DOUBLE)
set(WarpX_PARTICLE_PRECISION ${WarpX_PRECISION} CACHE STRING "Particle floating point precision (SINGLE/DOUBLE)")
set_property(CACHE WarpX_PARTICLE_PRECISION PROPERTY STRINGS ${WarpX_PARTICLE_PRECISION_VALUES})
if(NOT WarpX_PARTICLE_PRECISION IN_LIST WarpX_PARTICLE_PRECISION_VALUES)
    message(FATAL_ERROR "WarpX_PARTICLE_PRECISION (${WarpX_PARTICLE_PRECISION}) must be one of ${WarpX_PARTICLE_PRECISION_VALUES}")
endif()

set(WarpX_QED_TABLES_GEN_OMP_VALUES AUTO ON OFF)
set(WarpX_QED_TABLES_GEN_OMP AUTO CACHE STRING "Enables OpenMP support for QED lookup tables generation (AUTO/ON/OFF)")
set_property(CACHE WarpX_QED_TABLES_GEN_OMP PROPERTY STRINGS ${WarpX_QED_TABLES_GEN_OMP_VALUES})
if(NOT WarpX_QED_TABLES_GEN_OMP IN_LIST WarpX_QED_TABLES_GEN_OMP_VALUES)
    message(FATAL_ERROR "WarpX_QED_TABLES_GEN_OMP (${WarpX_QED_TABLES_GEN_OMP}) must be one of ${WarpX_QED_TABLES_GEN_OMP_VALUES}")
endif()

set(WarpX_COMPUTE_VALUES NOACC OMP CUDA SYCL HIP)
set(WarpX_COMPUTE OMP CACHE STRING "On-node, accelerated computing backend (NOACC/OMP/CUDA/SYCL/HIP)")
set_property(CACHE WarpX_COMPUTE PROPERTY STRINGS ${WarpX_COMPUTE_VALUES})
if(NOT WarpX_COMPUTE IN_LIST WarpX_COMPUTE_VALUES)
    message(FATAL_ERROR "WarpX_COMPUTE (${WarpX_COMPUTE}) must be one of ${WarpX_COMPUTE_VALUES}")
endif()

option(WarpX_MPI_THREAD_MULTIPLE "MPI thread-multiple support, i.e. for async_io" ON)
mark_as_advanced(WarpX_MPI_THREAD_MULTIPLE)

option(WarpX_amrex_internal                    "Download & build AMReX" ON)

# change the default build type to Release (or RelWithDebInfo) instead of Debug
set_default_build_type("Release")

# Option to enable interprocedural optimization
# (also know as "link-time optimization" or "whole program optimization")
set(_WarpX_IPO_DEFAULT OFF)
set(_WarpX_PYTHON_IPO_DEFAULT ON)
if(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(_WarpX_IPO_DEFAULT ${CMAKE_INTERPROCEDURAL_OPTIMIZATION})
    set(_WarpX_PYTHON_IPO_DEFAULT ${CMAKE_INTERPROCEDURAL_OPTIMIZATION})
endif()
option(WarpX_IPO
    "Compile WarpX with interprocedural optimization (will take more time)"
    ${_WarpX_IPO_DEFAULT}
)
option(WarpX_PYTHON_IPO
    "Compile Python bindings with interprocedural optimization (IPO) / link-time optimization (LTO)"
    ${_WarpX_PYTHON_IPO_DEFAULT}
)

set(pyWarpX_VERSION_INFO "" CACHE STRING
    "PEP-440 conformant version (set by setup.py)")

# enforce consistency of dependent options
if(WarpX_APP OR WarpX_PYTHON)
    set(WarpX_LIB ON CACHE STRING "Build WarpX as a library" FORCE)
endif()

# deprecated options: transition phase

if(DEFINED WarpX_PSATD)
    message(WARNING "CMake option WarpX_PSATD is deprecated. Use WarpX_FFT instead.\n"
                    "Overwriting WarpX_FFT with '${WarpX_PSATD}' because WarpX_PSATD was set.")
    set(WarpX_FFT ${WarpX_PSATD} CACHE STRING "FFT-based solvers" FORCE)
endif()

# note: we could skip this if we solely build WarpX_APP, but if we build a
# shared WarpX library or a third party, like ImpactX, uses ablastr in a
# shared library (e.g., for Python bindings), then we need relocatable code.
option(ABLASTR_POSITION_INDEPENDENT_CODE
       "Build ABLASTR with position independent code" ON)
mark_as_advanced(ABLASTR_POSITION_INDEPENDENT_CODE)

option(ABLASTR_FFT "compile AnyFFT wrappers" ${WarpX_FFT})
if(WarpX_FFT)
    set(ABLASTR_FFT ON CACHE STRING "FFT-based solvers" FORCE)
endif()

# Define the variable BUILD_TESTING (ON by default),
# include CDash dashboard testing module
include(CTest)


# Dependencies ################################################################
#

# AMReX
#   builds AMReX from source (default) or finds an existing install
set(WarpX_amrex_dim ${WarpX_DIMS})  # RZ is AMReX 2D
list(TRANSFORM WarpX_amrex_dim REPLACE RZ 2)
list(REMOVE_DUPLICATES WarpX_amrex_dim)

include(${WarpX_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)
foreach(D IN LISTS WarpX_amrex_dim)
    #   suppress warnings in AMReX headers (use -isystem instead of -I)
    warpx_make_third_party_includes_system(AMReX::amrex_${D}d amrex_${D}d)
endforeach()
# For <AMReX_buildInfo.H>
include(AMReXBuildInfo)

# PICSAR
#   builds PICSAR from source
include(${WarpX_SOURCE_DIR}/cmake/dependencies/PICSAR.cmake)

# openPMD
#   builds openPMD-api from source (default) or finds an existing install
include(${WarpX_SOURCE_DIR}/cmake/dependencies/openPMD.cmake)

# PSATD
include(${WarpX_SOURCE_DIR}/cmake/dependencies/FFT.cmake)
if(WarpX_FFT)
    # BLASPP and LAPACKPP
    if(RZ IN_LIST WarpX_DIMS)
        find_package(blaspp CONFIG REQUIRED)
        find_package(lapackpp CONFIG REQUIRED)
        find_package(OpenMP REQUIRED)  # pulled by the two above
    endif()
endif()

# Python
if(WarpX_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

    # default installation directories: Python
    warpx_set_default_install_dirs_python()

    # pybind11
    #   builds pybind11 from git (default), form local source or
    #   finds an existing install
    include(${WarpX_SOURCE_DIR}/cmake/dependencies/pybind11.cmake)

    # pyAMReX
    include(${WarpX_SOURCE_DIR}/cmake/dependencies/pyAMReX.cmake)
endif()


# Targets #####################################################################
#
include(GenerateExportHeader)
set(_ALL_TARGETS)
foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})

    # ABLASTR library
    add_library(ablastr_${SD})
    set(_BUILDINFO_SRC ablastr_${SD})
    list(APPEND _ALL_TARGETS ablastr_${SD})
    add_library(WarpX::ablastr_${SD} ALIAS ablastr_${SD})

    # link into a library (default: static)
    if(WarpX_LIB)
        add_library(lib_${SD})
        add_library(WarpX::lib_${SD} ALIAS lib_${SD})
        target_link_libraries(lib_${SD} PUBLIC ablastr_${SD})
        set(_BUILDINFO_SRC lib_${SD})
        list(APPEND _ALL_TARGETS lib_${SD})

        set_target_properties(lib_${SD} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            WINDOWS_EXPORT_ALL_SYMBOLS ON
        )
    endif()

    # executable application
    #   note: we currently avoid a dependency on a core library
    #         for simpler usage, but could make this an option
    if(WarpX_APP)
        add_executable(app_${SD})
        add_executable(WarpX::app_${SD} ALIAS app_${SD})
        target_link_libraries(app_${SD} PRIVATE lib_${SD})
        set(_BUILDINFO_SRC app_${SD})
        list(APPEND _ALL_TARGETS app_${SD})
    endif()

    if(WarpX_PYTHON OR (WarpX_LIB AND BUILD_SHARED_LIBS))
        set(ABLASTR_POSITION_INDEPENDENT_CODE ON CACHE BOOL
            "Build ABLASTR with position independent code" FORCE)
    endif()

    # ABLASTR library (static or shared)
    set_target_properties(ablastr_${SD} PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
    if(ABLASTR_POSITION_INDEPENDENT_CODE)
        set_target_properties(ablastr_${SD} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
        )
    endif()

    # own headers
    target_include_directories(ablastr_${SD} PUBLIC
        # future: own directory root
        $<BUILD_INTERFACE:${WarpX_SOURCE_DIR}/Source>
        $<BUILD_INTERFACE:${WarpX_BINARY_DIR}/Source>
    )
    if(WarpX_LIB)
        target_include_directories(lib_${SD} PUBLIC
            $<BUILD_INTERFACE:${WarpX_SOURCE_DIR}/Source>
            $<BUILD_INTERFACE:${WarpX_BINARY_DIR}/Source>
        )
    endif()

    # build Python module (this is always a shared library)
    if(WarpX_PYTHON)
        add_library(pyWarpX_${SD} MODULE Source/Python/pyWarpX.cpp)
        add_library(WarpX::pyWarpX_${SD} ALIAS pyWarpX_${SD})
        target_link_libraries(pyWarpX_${SD} PUBLIC lib_${SD})
        set(_BUILDINFO_SRC pyWarpX_${SD})
        list(APPEND _ALL_TARGETS pyWarpX_${SD})

        # set Python module properties
        set_target_properties(pyWarpX_${SD} PROPERTIES
            # hide symbols for combining multiple pybind11 modules downstream & for
            # reduced binary size
            CXX_VISIBILITY_PRESET "hidden"
            CUDA_VISIBILITY_PRESET "hidden"
            # name of the pybind-generated python module, which is wrapped in another
            # fluffy front-end modules, so we can extend it with pure Python
            ARCHIVE_OUTPUT_NAME warpx_pybind_${SD}
            LIBRARY_OUTPUT_NAME warpx_pybind_${SD}
            # build output directories - mainly set to run tests from CMake & IDEs
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
            PDB_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
            COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
        )
        get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        if(isMultiConfig)
            foreach(CFG IN LISTS CMAKE_CONFIGURATION_TYPES)
                string(TOUPPER "${CFG}" CFG_UPPER)
                set_target_properties(pyWarpX_${SD} PROPERTIES
                    # build output directories - mainly set to run tests from CMake & IDEs
                    # note: same as above, but for Multi-Config generators
                    ARCHIVE_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
                    LIBRARY_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
                    RUNTIME_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
                    PDB_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
                    COMPILE_PDB_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/pywarpx
                )
            endforeach()
        endif()
        if(EMSCRIPTEN)
            set_target_properties(pyWarpX_${SD} PROPERTIES
                PREFIX "")
        else()
            pybind11_extension(pyWarpX_${SD})
        endif()
        if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
            pybind11_strip(pyWarpX_${SD})
        endif()
    endif()

    if(WarpX_LIB)
        # if we include <AMReX_buildInfo.H> we will need to call:
        generate_buildinfo(${_BUILDINFO_SRC} "${WarpX_SOURCE_DIR}")
        target_link_libraries(lib_${SD} PRIVATE buildInfo::${_BUILDINFO_SRC})
        unset(_BUILDINFO_SRC)

        # add sources
        target_sources(lib_${SD} PRIVATE Source/WarpX.cpp)
    endif()

    # add sources
    if(WarpX_APP)
        target_sources(app_${SD} PRIVATE Source/main.cpp)
    endif()
endforeach()

# Headers controlling symbol visibility (for Windows)
# Note: once WarpX gets rid of all its globals, we will not need this.
#   - extern ... functions
#   - static class member variables
generate_export_header(ablastr_${WarpX_DIMS_LAST}
    BASE_NAME ablastr
    EXPORT_FILE_NAME Source/ablastr/export.H)
if(WarpX_LIB)
    generate_export_header(lib_${WarpX_DIMS_LAST}
        BASE_NAME warpx
        EXPORT_FILE_NAME Source/Utils/export.H)
endif()
# At build-time, we might need to set WarpX_<last>_EXPORTS and ablastr_<last>_EXPORTS on all targets
foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_compile_definitions(ablastr_${SD} PRIVATE ablastr_${WarpX_DIMS_LAST}_EXPORTS)
    if(WarpX_LIB)
        target_compile_definitions(lib_${SD} PRIVATE lib_${WarpX_DIMS_LAST}_EXPORTS)  # note: some collision risk here
    endif()
    # Static libs: WARPX_STATIC_DEFINE and ABLASTR_STATIC_DEFINE during build time
    # note: using a static lib (lib_${SD}) in a .dll (pyWarpX_${SD}) still needs the exports for
    #       dllimport at DLL build time.
    #       https://cmake.org/cmake/help/latest/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(ablastr_${SD} PRIVATE ABLASTR_STATIC_DEFINE)
        if(WarpX_LIB)
            target_compile_definitions(lib_${SD} PRIVATE WARPX_STATIC_DEFINE)
        endif()
    endif()
endforeach()

if(WarpX_PYTHON)
    # copy PICMI and other Python scripts to build directory
    add_custom_command(TARGET pyWarpX_${WarpX_DIMS_LAST} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${WarpX_SOURCE_DIR}/Python/pywarpx
        $<TARGET_FILE_DIR:pyWarpX_${WarpX_DIMS_LAST}>
    )
endif()

add_subdirectory(Source/ablastr)

if(WarpX_LIB)
    add_subdirectory(Source/AcceleratorLattice)
    add_subdirectory(Source/BoundaryConditions)
    add_subdirectory(Source/Diagnostics)
    add_subdirectory(Source/EmbeddedBoundary)
    add_subdirectory(Source/Evolve)
    add_subdirectory(Source/FieldSolver)
    add_subdirectory(Source/Filter)
    add_subdirectory(Source/Fluids)
    add_subdirectory(Source/Initialization)
    add_subdirectory(Source/Laser)
    add_subdirectory(Source/Parallelization)
    add_subdirectory(Source/Particles)
    add_subdirectory(Source/Python)
    add_subdirectory(Source/Utils)
endif()
if(WarpX_QED_TOOLS)
    add_subdirectory(Tools/QedTablesUtils)
endif()

# Interprocedural optimization (IPO) / Link-Time Optimization (LTO)
if(WarpX_IPO)
    warpx_enable_IPO("${_ALL_TARGETS}")
endif()

# link dependencies
foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    if(D STREQUAL "RZ")
        target_link_libraries(ablastr_${SD} PUBLIC WarpX::thirdparty::amrex_2d)
    else()
        target_link_libraries(ablastr_${SD} PUBLIC WarpX::thirdparty::amrex_${D}d)
    endif()

    if(ABLASTR_FFT)
        target_link_libraries(ablastr_${SD} PUBLIC WarpX::thirdparty::FFT)
        if(D STREQUAL "RZ")
            target_link_libraries(ablastr_${SD} PUBLIC blaspp)
            target_link_libraries(ablastr_${SD} PUBLIC lapackpp)

            # BLAS++ forgets to declare cuBLAS and cudaRT dependencies
            if(WarpX_COMPUTE STREQUAL CUDA)
                find_package(CUDAToolkit REQUIRED)
                target_link_libraries(ablastr_${SD} PUBLIC CUDA::cudart CUDA::cublas)
            endif()
        endif()
    endif()

    if(WarpX_PYTHON)
        target_link_libraries(pyWarpX_${SD} PRIVATE pybind11::module pybind11::windows_extras)
        if(WarpX_PYTHON_IPO)
            if(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
                warpx_enable_IPO(pyWarpX_${SD})
            else()
                # conditionally defined target in pybind11
                # https://github.com/pybind/pybind11/blob/v2.13.0/tools/pybind11Common.cmake#L407-L413
                target_link_libraries(pyWarpX_${SD} PRIVATE pybind11::lto)
            endif()
        endif()
    endif()

    if(WarpX_OPENPMD)
        target_link_libraries(ablastr_${SD} PUBLIC openPMD::openPMD)
    endif()

    if(WarpX_QED)
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_QED)
        if(WarpX_QED_TABLE_GEN)
            target_compile_definitions(ablastr_${SD} PUBLIC WARPX_QED_TABLE_GEN)
        endif()
        target_link_libraries(ablastr_${SD} PUBLIC PXRMP_QED::PXRMP_QED)
    endif()
endforeach()

# C++ properties: at least a C++17 capable compiler is needed
if(WarpX_COMPUTE STREQUAL CUDA)
    # AMReX helper function: propagate CUDA specific target & source properties
    foreach(warpx_tgt IN LISTS _ALL_TARGETS)
        setup_target_for_cuda_compilation(${warpx_tgt})
    endforeach()
    foreach(warpx_tgt IN LISTS _ALL_TARGETS)
        target_compile_features(${warpx_tgt} PUBLIC cuda_std_17)
    endforeach()
    set_target_properties(${_ALL_TARGETS} PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
    )
else()
    foreach(warpx_tgt IN LISTS _ALL_TARGETS)
        target_compile_features(${warpx_tgt} PUBLIC cxx_std_17)
    endforeach()
    set_target_properties(${_ALL_TARGETS} PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
    )
endif()

foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})

    # fancy binary name for build variants
    set_warpx_binary_name(${D})
endforeach()


# Defines #####################################################################
#
foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    if(D STREQUAL 3)
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_DIM_3D WARPX_ZINDEX=2)
    elseif(D STREQUAL 2)
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_DIM_XZ WARPX_ZINDEX=1)
    elseif(D STREQUAL 1)
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_DIM_1D_Z WARPX_ZINDEX=0)
    elseif(D STREQUAL "RZ")
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_DIM_RZ WARPX_ZINDEX=1)
    endif()

    if(WarpX_OPENPMD)
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_USE_OPENPMD)
    endif()

    if(WarpX_QED)
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_QED)
        if(WarpX_QED_TABLE_GEN)
            target_compile_definitions(ablastr_${SD} PUBLIC WarpX_QED_TABLE_GEN)
        endif()
    endif()

    if(WarpX_FFT)
        target_compile_definitions(ablastr_${SD} PUBLIC WARPX_USE_FFT)
    endif()
    if(ABLASTR_FFT)
        # We need to enable FFT support in ABLASTR for PSATD solver
        target_compile_definitions(ablastr_${SD} PUBLIC ABLASTR_USE_FFT)
    endif()

    if(WarpX_PYTHON AND pyWarpX_VERSION_INFO)
        # for module __version__
        target_compile_definitions(pyWarpX_${SD} PRIVATE
            PYWARPX_VERSION_INFO=${pyWarpX_VERSION_INFO})
    endif()

    # <cmath>: M_PI
    if(WIN32)
        target_compile_definitions(ablastr_${SD} PUBLIC _USE_MATH_DEFINES)
    endif()

    # Windows DLLs and Global Symbols
    # https://stackoverflow.com/questions/54560832/cmake-windows-export-all-symbols-does-not-cover-global-variables/54568678#54568678
    #
    if(WIN32 AND BUILD_SHARED_LIBS)
        set(ABLASTR_IS_DLL ON)
        target_compile_definitions(ablastr_${SD} PRIVATE ABLASTR_IS_DLL_BUILDING)
    endif()
    if(WIN32 AND WarpX_LIB AND BUILD_SHARED_LIBS)
        set(ABLASTR_IS_DLL ON)
        target_compile_definitions(lib_${SD} PRIVATE ABLASTR_IS_DLL_BUILDING)
    endif()
    if(WIN32 AND WarpX_PYTHON)
        set(ABLASTR_IS_DLL ON)
        target_compile_definitions(pyWarpX_${SD} PRIVATE ABLASTR_IS_DLL_BUILDING)
    endif()
endforeach()


# Warnings ####################################################################
#
set_cxx_warnings()


# Generate Configuration and .pc Files ########################################
#
get_source_version(WarpX_${WarpX_DIMS_LAST} ${WarpX_SOURCE_DIR})
set(WarpX_GIT_VERSION ${WarpX_${WarpX_DIMS_LAST}_GIT_VERSION})
configure_file(
    ${WarpX_SOURCE_DIR}/Source/Utils/WarpXVersion.H.in
    ${WarpX_BINARY_DIR}/Source/Utils/WarpXVersion.H
    @ONLY
)

# these files are used if WarpX is installed and picked up by a downstream
# project (not needed yet)

#include(CMakePackageConfigHelpers)
#write_basic_package_version_file("WarpXConfigVersion.cmake"
#    VERSION ${WarpX_VERSION}
#    COMPATIBILITY SameMajorVersion
#)


# Installs ####################################################################
#
# headers, libraries and executables
set(WarpX_INSTALL_TARGET_NAMES)
foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    list(APPEND WarpX_INSTALL_TARGET_NAMES ablastr_${SD})
    if(WarpX_APP)
        list(APPEND WarpX_INSTALL_TARGET_NAMES app_${SD})
    endif()
    if(WarpX_LIB)
        list(APPEND WarpX_INSTALL_TARGET_NAMES lib_${SD})
    endif()
    if(WarpX_PYTHON)
        list(APPEND WarpX_INSTALL_TARGET_NAMES pyWarpX_${SD})
    endif()

    install(TARGETS ${WarpX_INSTALL_TARGET_NAMES}
        EXPORT WarpXTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # simplified library alias
    # this is currently expected by Python bindings
    if(WarpX_LIB)
        if(IS_ABSOLUTE ${CMAKE_INSTALL_LIBDIR})
            set(ABS_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
        else()
            set(ABS_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
        endif()
        # escape spaces for generated cmake_install.cmake file
        file(TO_CMAKE_PATH "${ABS_INSTALL_LIB_DIR}" ABS_INSTALL_LIB_DIR)

        install(CODE "file(CREATE_LINK
            $<TARGET_FILE_NAME:lib_${SD}>
            \"${ABS_INSTALL_LIB_DIR}/libwarpx.${SD}$<TARGET_FILE_SUFFIX:lib_${SD}>\"
            COPY_ON_ERROR SYMBOLIC)")
    endif()

    # WarpX
    # Utils/WarpXVersion.H

    # ABLASTR
    # export.H

    # CMake package file for find_package(WarpX::WarpX) in depending projects
    #install(EXPORT WarpXTargets
    #    FILE WarpXTargets.cmake
    #    NAMESPACE WarpX::
    #    DESTINATION ${WarpX_INSTALL_CMAKEDIR}
    #)
    #install(
    #    FILES
    #        ${WarpX_BINARY_DIR}/WarpXConfig.cmake
    #        ${WarpX_BINARY_DIR}/WarpXConfigVersion.cmake
    #    DESTINATION ${WarpX_INSTALL_CMAKEDIR}
    #)
endforeach()


# pip helpers for the pywarpx package #########################################
#
if(WarpX_PYTHON)
    set(PY_PIP_OPTIONS "-v" CACHE STRING
        "Additional parameters to pass to `pip` as ; separated list")
    set(PY_PIP_INSTALL_OPTIONS "" CACHE STRING
        "Additional parameters to pass to `pip install` as ; separated list")

    # ensure all targets are built before we package them in a wheel
    set(pyWarpX_INSTALL_TARGET_NAMES)
    foreach(D IN LISTS WarpX_DIMS)
        warpx_set_suffix_dims(SD ${D})
        list(APPEND pyWarpX_INSTALL_TARGET_NAMES pyWarpX_${SD})
    endforeach()

    # add a prefix to custom targets so we do not collide if used as a subproject
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(_WarpX_CUSTOM_TARGET_PREFIX_DEFAULT "")
    else()
        set(_WarpX_CUSTOM_TARGET_PREFIX_DEFAULT "warpx_")
    endif()
    set(WarpX_CUSTOM_TARGET_PREFIX "${_WarpX_CUSTOM_TARGET_PREFIX_DEFAULT}"
            CACHE STRING "Prefix for custom targets")

    # build the wheel by re-using the python module that we build
    add_custom_target(${WarpX_CUSTOM_TARGET_PREFIX}pip_wheel
        ${CMAKE_COMMAND} -E rm -f -r warpx-whl
        COMMAND
            ${CMAKE_COMMAND} -E env PYWARPX_LIB_DIR=$<TARGET_FILE_DIR:pyWarpX_${WarpX_DIMS_LAST}>
                ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} wheel --no-build-isolation --no-deps --wheel-dir=warpx-whl "${WarpX_SOURCE_DIR}"
        COMMAND_EXPAND_LISTS VERBATIM
        WORKING_DIRECTORY
            ${WarpX_BINARY_DIR}
        DEPENDS
            ${pyWarpX_INSTALL_TARGET_NAMES}
    )

    # this will also upgrade/downgrade dependencies, e.g., when the version of picmistandard changes
    if(WarpX_MPI)
        set(pyWarpX_REQUIREMENT_FILE "requirements_mpi.txt")
    else()
        set(pyWarpX_REQUIREMENT_FILE "requirements.txt")
    endif()
    add_custom_target(${WarpX_CUSTOM_TARGET_PREFIX}pip_install_requirements
        ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install ${PY_PIP_INSTALL_OPTIONS} -r "${WarpX_SOURCE_DIR}/${pyWarpX_REQUIREMENT_FILE}"
        COMMAND_EXPAND_LISTS VERBATIM
        WORKING_DIRECTORY
            ${WarpX_BINARY_DIR}
    )

    # if we do a superbuild, make sure we install pyAMReX via its custom install
    # target
    set(_EXTRA_INSTALL_DEPENDS)
    if(WarpX_pyamrex_internal OR WarpX_pyamrex_src)
        set(_EXTRA_INSTALL_DEPENDS pyamrex_pip_install)
    endif()

    # We force-install because in development, it is likely that the version of
    # the package does not change, but the code did change. We need --no-deps,
    # because otherwise pip would also force reinstall all dependencies.
    add_custom_target(${WarpX_CUSTOM_TARGET_PREFIX}pip_install
        ${CMAKE_COMMAND} -E env WARPX_MPI=${WarpX_MPI}
            ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install --force-reinstall --no-index --no-deps ${PY_PIP_INSTALL_OPTIONS} --find-links=warpx-whl pywarpx
        COMMAND_EXPAND_LISTS VERBATIM
        WORKING_DIRECTORY
            ${WarpX_BINARY_DIR}
        DEPENDS
            pyWarpX_${WarpX_DIMS_LAST}
            ${WarpX_CUSTOM_TARGET_PREFIX}pip_wheel
            ${WarpX_CUSTOM_TARGET_PREFIX}pip_install_requirements
            ${_EXTRA_INSTALL_DEPENDS}
    )

    # this is for package managers only
    add_custom_target(${WarpX_CUSTOM_TARGET_PREFIX}pip_install_nodeps
        ${CMAKE_COMMAND} -E env WARPX_MPI=${WarpX_MPI}
            ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install --force-reinstall --no-index --no-deps ${PY_PIP_INSTALL_OPTIONS} --find-links=warpx-whl pywarpx
        COMMAND_EXPAND_LISTS VERBATIM
        WORKING_DIRECTORY
            ${WarpX_BINARY_DIR}
        DEPENDS
            pyWarpX_${WarpX_DIMS_LAST}
            ${WarpX_CUSTOM_TARGET_PREFIX}pip_wheel
    )
endif()


# Tests #######################################################################
#
if(BUILD_TESTING)
    enable_testing()
    if(WarpX_APP OR WarpX_PYTHON)
        add_subdirectory(Examples)
    endif()
endif()


# Status Summary for Build Options ############################################
#
warpx_print_summary()
