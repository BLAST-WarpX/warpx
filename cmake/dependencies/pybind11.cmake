function(find_pybind11)
    if(WarpX_pybind11_src)
        message(STATUS "Compiling local pybind11 ...")
        message(STATUS "pybind11 source path: ${WarpX_pybind11_src}")
        if(NOT IS_DIRECTORY ${WarpX_pybind11_src})
            message(FATAL_ERROR "Specified directory WarpX_pybind11_src='${WarpX_pybind11_src}' does not exist!")
        endif()
    elseif(WarpX_pybind11_internal)
        message(STATUS "Downloading pybind11 ...")
        message(STATUS "pybind11 repository: ${WarpX_pybind11_repo} (${WarpX_pybind11_branch})")
        include(FetchContent)
    endif()

    # rely on our find_package(Python ...) call
    # https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-cmake
    set(PYBIND11_FINDPYTHON ON)

    if(WarpX_pybind11_internal OR WarpX_pybind11_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        if(WarpX_pybind11_src)
            add_subdirectory(${WarpX_pybind11_src} _deps/localpybind11-build/)
        else()
            FetchContent_Declare(fetchedpybind11
                GIT_REPOSITORY ${WarpX_pybind11_repo}
                GIT_TAG        ${WarpX_pybind11_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedpybind11)

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDpybind11)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDpybind11)
        endif()
    else()
        find_package(pybind11 2.13.0 CONFIG REQUIRED)
        message(STATUS "pybind11: Found version '${pybind11_VERSION}'")
    endif()
endfunction()

# local source-tree
set(WarpX_pybind11_src ""
    CACHE PATH
    "Local path to pybind11 source directory (preferred if set)")

# Git fetcher
option(WarpX_pybind11_internal "Download & build pybind11" ON)
set(WarpX_pybind11_repo "https://github.com/pybind/pybind11.git"
    CACHE STRING
    "Repository URI to pull and build pybind11 from if(WarpX_pybind11_internal)")
set(WarpX_pybind11_branch "v2.13.6"
    CACHE STRING
    "Repository branch for WarpX_pybind11_repo if(WarpX_pybind11_internal)")

find_pybind11()
