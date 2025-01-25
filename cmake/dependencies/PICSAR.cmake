function(find_picsar)
    if(WarpX_picsar_src)
        message(STATUS "Compiling local PICSAR ...")
        message(STATUS "PICSAR source path: ${WarpX_picsar_src}")
        if(NOT IS_DIRECTORY ${WarpX_picsar_src})
            message(FATAL_ERROR "Specified directory WarpX_picsar_src='${WarpX_picsar_src}' does not exist!")
        endif()
    elseif(WarpX_picsar_internal)
        message(STATUS "Downloading PICSAR ...")
        message(STATUS "PICSAR repository: ${WarpX_picsar_repo} (${WarpX_picsar_branch})")
        include(FetchContent)
    endif()
    if(WarpX_picsar_internal OR WarpX_picsar_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # Enable or disable QED lookup tables generation
        if(WarpX_QED_TABLE_GEN OR WarpX_QED_TOOLS)
            set(PXRMP_QED_TABLEGEN ON CACHE INTERNAL "")
        else()
            set(PXRMP_QED_TABLEGEN OFF CACHE INTERNAL "")
        endif()

        # Enable or disable OpenMP for lookup tables generation
        if(PXRMP_QED_TABLEGEN)
            if(WarpX_QED_TABLES_GEN_OMP STREQUAL AUTO)
                find_package(OpenMP REQUIRED CXX)
                if(OpenMP_CXX_FOUND)
                    set(PXRMP_QED_OMP ON CACHE INTERNAL "")
                else()
                    set(PXRMP_QED_OMP OFF CACHE INTERNAL "")
                endif()
            elseif(WarpX_QED_TABLES_GEN_OMP STREQUAL ON)
                    set(PXRMP_QED_OMP ON CACHE INTERNAL "")
            else()
                    set(PXRMP_QED_OMP OFF CACHE INTERNAL "")
            endif()
        else()
            set(PXRMP_QED_OMP OFF CACHE INTERNAL "")
        endif()

        # Always disable tests
        set (PXRMP_QED_TEST OFF CACHE INTERNAL "")

        if(WarpX_COMPUTE STREQUAL SYCL)
            set (PXRMP_DPCPP_FIX ON CACHE INTERNAL "")
        endif()

        if(WarpX_picsar_src)
            add_subdirectory(
                ${WarpX_picsar_src}/multi_physics/QED
                _deps/localpicsar-build/
            )
            get_source_version(PXRMP_QED ${WarpX_picsar_src})
        else()
            FetchContent_Declare(fetchedpicsar
                GIT_REPOSITORY  ${WarpX_picsar_repo}
                GIT_TAG         ${WarpX_picsar_branch}
                BUILD_IN_SOURCE 0
                SOURCE_SUBDIR   multi_physics/QED
            )
            FetchContent_MakeAvailable(fetchedpicsar)

            get_source_version(PXRMP_QED ${fetchedpicsar_SOURCE_DIR})
            if(NOT PXRMP_QED_GIT_VERSION)
                set(PXRMP_QED_GIT_VERSION "${WarpX_picsar_branch}" CACHE INTERNAL "")
            endif()

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDPICSAR)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDPICSAR)
        endif()

        # PICSAR options not relevant to WarpX users
        mark_as_advanced(DIM)
        mark_as_advanced(USE_XSDK_DEFAULTS)
        mark_as_advanced(PXRMP_QED_TABLEGEN)
        mark_as_advanced(PXRMP_QED_OMP)
        mark_as_advanced(PXRMP_QED_TEST)
        mark_as_advanced(PXRMP_BOOST_TEST_DYN_LINK)
        mark_as_advanced(PXRMP_DPCPP_FIX)


        # PICSAR_VERSION: not yet defined
        #message(STATUS "PICSAR: Using version '${PICSAR_VERSION}'")
    else()
        # not supported by PICSAR (yet)
        #find_package(PICSAR 24.09 CONFIG REQUIRED QED)
        #message(STATUS "PICSAR: Found version '${PICSAR_VERSION}'")
        message(FATAL_ERROR "PICSAR: Cannot be used as externally installed "
            "library yet. "
            "Note that you can point to an external source tree with "
            "-DWarpX_picsar_src=<path>"
        )
    endif()
endfunction()

if(WarpX_QED)
    # local source-tree
    set(WarpX_picsar_src ""
        CACHE PATH
        "Local path to PICSAR source directory (preferred if set)")

    # Git fetcher
    option(WarpX_picsar_internal   "Download & build PICSAR" ON)
    set(WarpX_picsar_repo "https://github.com/ECP-WarpX/picsar.git"
        CACHE STRING
        "Repository URI to pull and build PICSAR from if(WarpX_picsar_internal)")
    set(WarpX_picsar_branch "47b393993f860943e387b4b5d79407ee7f52d1ab"
        CACHE STRING
        "Repository branch for WarpX_picsar_repo if(WarpX_picsar_internal)")

    cmake_dependent_option(WarpX_QED_TABLE_GEN "generate QED lookup tables (requires boost)"
        ON "WarpX_QED" OFF)

    find_picsar()
endif()
