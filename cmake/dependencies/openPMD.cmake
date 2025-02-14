function(find_openpmd)
    if(WarpX_openpmd_src)
        message(STATUS "Compiling local openPMD-api ...")
        message(STATUS "openPMD-api source path: ${WarpX_openpmd_src}")
        if(NOT IS_DIRECTORY ${WarpX_openpmd_src})
            message(FATAL_ERROR "Specified directory WarpX_openpmd_src='${WarpX_openpmd_src}' does not exist!")
        endif()
    elseif(WarpX_openpmd_internal)
        message(STATUS "Downloading openPMD-api ...")
        message(STATUS "openPMD-api repository: ${WarpX_openpmd_repo} (${WarpX_openpmd_branch})")
        include(FetchContent)
    endif()
    if(WarpX_openpmd_internal OR WarpX_openpmd_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://openpmd-api.readthedocs.io/en/0.16.1/dev/buildoptions.html
        set(openPMD_USE_ADIOS1      OFF           CACHE INTERNAL "")
        set(openPMD_USE_MPI         ${WarpX_MPI}  CACHE INTERNAL "")
        set(openPMD_USE_PYTHON      OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_CLI_TOOLS OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_EXAMPLES  OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_TESTING   OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_SHARED_LIBS  ${BUILD_SHARED_LIBS} CACHE INTERNAL "")
        set(openPMD_INSTALL  ${openPMD_BUILD_SHARED_LIBS} CACHE INTERNAL "")
        # openPMD_INSTALL_RPATH

        if(WarpX_openpmd_src)
            add_subdirectory(${WarpX_openpmd_src} _deps/localopenpmd-build/)
        else()
            FetchContent_Declare(fetchedopenpmd
                GIT_REPOSITORY ${WarpX_openpmd_repo}
                GIT_TAG        ${WarpX_openpmd_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedopenpmd)

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDOPENPMD)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDOPENPMD)
        endif()

        # openPMD options not relevant to WarpX users
        mark_as_advanced(openPMD_HAVE_PKGCONFIG)
        mark_as_advanced(openPMD_INSTALL_RPATH)
        mark_as_advanced(openPMD_USE_CUDA_EXAMPLES)
        mark_as_advanced(openPMD_USE_INTERNAL_CATCH)
        mark_as_advanced(openPMD_USE_INTERNAL_PYBIND11)
        mark_as_advanced(openPMD_USE_INTERNAL_JSON)
        mark_as_advanced(openPMD_USE_INTERNAL_TOML11)
        mark_as_advanced(openPMD_USE_INVASIVE_TESTS)
        mark_as_advanced(openPMD_USE_VERIFY)
        mark_as_advanced(ADIOS2_DIR)
        mark_as_advanced(ADIOS_CONFIG)
        mark_as_advanced(HDF5_DIR)
        mark_as_advanced(HDF5_C_LIBRARY_dl)
        mark_as_advanced(HDF5_C_LIBRARY_hdf5)
        mark_as_advanced(HDF5_C_LIBRARY_m)
        mark_as_advanced(HDF5_C_LIBRARY_z)
        mark_as_advanced(JSON_ImplicitConversions)
        mark_as_advanced(JSON_MultipleHeaders)
        mark_as_advanced(toml11_BUILD_TEST)
        mark_as_advanced(toml11_TEST_WITH_ASAN)
        mark_as_advanced(toml11_TEST_WITH_UBSAN)
    else()
        if(WarpX_MPI)
            set(COMPONENT_WMPI MPI)
        else()
            set(COMPONENT_WMPI NOMPI)
        endif()
        find_package(openPMD 0.16.1 CONFIG REQUIRED COMPONENTS ${COMPONENT_WMPI})
        message(STATUS "openPMD-api: Found version '${openPMD_VERSION}'")
    endif()
endfunction()

if(WarpX_OPENPMD)
    # local source-tree
    set(WarpX_openpmd_src ""
        CACHE PATH
        "Local path to openPMD-api source directory (preferred if set)")

    # Git fetcher
    option(WarpX_openpmd_internal   "Download & build openPMD-api" ON)
    set(WarpX_openpmd_repo "https://github.com/openPMD/openPMD-api.git"
        CACHE STRING
        "Repository URI to pull and build openPMD-api from if(WarpX_openpmd_internal)")
    set(WarpX_openpmd_branch "0.16.1"
        CACHE STRING
        "Repository branch for WarpX_openpmd_repo if(WarpX_openpmd_internal)")

    set(WarpX_HAVE_OPENPMD TRUE)
    find_openpmd()
endif()
