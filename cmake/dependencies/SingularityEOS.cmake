# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

function(find_singularity_eos)
    # Singularity-EOS changes the global BUILD_TESTING cache option in
    # submodule mode. Preserve the parent project's choice so adding this
    # dependency does not silently remove the WarpX test suite.
    set(_warpx_build_testing "${BUILD_TESTING}")

    if(WarpX_singularity_eos_src)
        if(NOT IS_DIRECTORY "${WarpX_singularity_eos_src}")
            message(FATAL_ERROR
                "WarpX_singularity_eos_src='${WarpX_singularity_eos_src}' "
                "does not exist")
        endif()
        message(STATUS
            "Compiling local Singularity-EOS: ${WarpX_singularity_eos_src}")
    elseif(WarpX_singularity_eos_internal)
        message(STATUS
            "Downloading Singularity-EOS ${WarpX_singularity_eos_branch}")
        include(FetchContent)
    else()
        find_package(singularity-eos CONFIG REQUIRED)
        if(SINGULARITY_USE_KOKKOS)
            message(FATAL_ERROR
                "The external Singularity-EOS package uses Kokkos, but the "
                "current WarpX table-EOS integration requires host-resident "
                "PORTABILITY_STRATEGY_NONE tables. Use the pinned internal "
                "dependency or a host-only external package.")
        endif()
        if(NOT SINGULARITY_USE_SPINER_WITH_HDF5)
            message(FATAL_ERROR
                "WarpX_SINGULARITY_EOS requires an external Singularity-EOS "
                "package built with SINGULARITY_USE_SPINER_WITH_HDF5=ON.")
        endif()
        return()
    endif()

    # File-backed Spiner tables require HDF5. Disable unrelated language
    # bindings, closures, tools and tests in this embedded build.
    set(SINGULARITY_USE_SPINER ON CACHE BOOL "" FORCE)
    set(SINGULARITY_USE_SPINER_WITH_HDF5 ON CACHE BOOL "" FORCE)
    set(SINGULARITY_USE_FORTRAN OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_USE_KOKKOS OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_USE_EOSPAC OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_BUILD_CLOSURE OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_BUILD_SESAME2SPINER OFF CACHE BOOL "" FORCE)
    set(SINGULARITY_BETTER_DEBUG_FLAGS OFF CACHE BOOL "" FORCE)

    if(WarpX_singularity_eos_src)
        add_subdirectory(
            "${WarpX_singularity_eos_src}"
            "${WarpX_BINARY_DIR}/_deps/localsingularityeos-build")
    else()
        FetchContent_Declare(fetchedsingularityeos
            GIT_REPOSITORY "${WarpX_singularity_eos_repo}"
            GIT_TAG "${WarpX_singularity_eos_branch}"
            GIT_SHALLOW FALSE
            GIT_SUBMODULES "utils/ports-of-call;utils/spiner"
            GIT_SUBMODULES_RECURSE TRUE
        )
        FetchContent_MakeAvailable(fetchedsingularityeos)
    endif()

    set(BUILD_TESTING "${_warpx_build_testing}" CACHE BOOL
        "Enable WarpX tests" FORCE)

    foreach(_singularity_target IN ITEMS
            singularity-eos_Interface singularity-utils spiner ports-of-call)
        if(TARGET ${_singularity_target})
            get_target_property(_singularity_includes
                ${_singularity_target} INTERFACE_INCLUDE_DIRECTORIES)
            if(_singularity_includes)
                set_property(TARGET ${_singularity_target} APPEND PROPERTY
                    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                    "${_singularity_includes}")
            endif()
        endif()
    endforeach()
endfunction()

if(WarpX_SINGULARITY_EOS)
    # The first integration milestone uses host-resident Spiner tables. CUDA,
    # HIP and SYCL support will be enabled only after allocator ownership is
    # covered by device tests.
    if(NOT WarpX_COMPUTE STREQUAL "OMP"
       AND NOT WarpX_COMPUTE STREQUAL "NOACC")
        message(FATAL_ERROR
            "WarpX_SINGULARITY_EOS currently supports WarpX_COMPUTE=OMP or "
            "NOACC. Other electron-thermodynamics backends remain portable "
            "to all WarpX compute backends.")
    endif()

    set(WarpX_singularity_eos_src "" CACHE PATH
        "Local Singularity-EOS source directory")
    option(WarpX_singularity_eos_internal
        "Download and build Singularity-EOS" ON)
    set(WarpX_singularity_eos_repo
        "https://github.com/lanl/singularity-eos.git" CACHE STRING
        "Singularity-EOS repository")

    file(READ "${WarpX_SOURCE_DIR}/dependencies.json" dependencies_data)
    string(JSON singularity_eos_version GET
        "${dependencies_data}" version_singularity_eos)
    string(JSON singularity_eos_commit GET
        "${dependencies_data}" commit_singularity_eos)
    set(WarpX_singularity_eos_branch
        "${singularity_eos_commit}" CACHE STRING
        "Pinned Singularity-EOS revision (release-${singularity_eos_version})")

    find_singularity_eos()
endif()
