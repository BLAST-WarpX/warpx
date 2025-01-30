function(find_pyamrex)
    if(WarpX_pyamrex_src)
        message(STATUS "Compiling local pyAMReX ...")
        message(STATUS "pyAMReX source path: ${WarpX_pyamrex_src}")
        if(NOT IS_DIRECTORY ${WarpX_pyamrex_src})
            message(FATAL_ERROR "Specified directory WarpX_pyamrex_src='${WarpX_pyamrex_src}' does not exist!")
        endif()
    elseif(WarpX_pyamrex_internal)
        message(STATUS "Downloading pyAMReX ...")
        message(STATUS "pyAMReX repository: ${WarpX_pyamrex_repo} (${WarpX_pyamrex_branch})")
        include(FetchContent)
    endif()

    # transitive control for AMReX & pybind11 superbuild
    #   note: if we do superbuilds, we want the same AMReX commit for
    #           AMReX->ABLASTR->WarpX and
    #           AMReX->pyAMReX->pyWarpX
    #   note: this is performed after we did the transitive logic control in
    #         ABLASTR.cmake
    set(pyAMReX_amrex_internal ${WarpX_amrex_internal} CACHE BOOL
        "Download & build AMReX" FORCE)
    set(pyAMReX_pybind11_internal ${WarpX_pybind11_internal} CACHE BOOL
        "Download & build AMReX" FORCE)

    if(WarpX_amrex_src)
        set(pyAMReX_amrex_src ${WarpX_amrex_src} CACHE PATH
            "Local path to AMReX source directory (preferred if set)" FORCE)
    elseif(WarpX_amrex_internal)
        if(WarpX_amrex_repo)
            set(pyAMReX_amrex_repo ${WarpX_amrex_repo} CACHE STRING
                "Repository URI to pull and build AMReX from if(WarpX_amrex_internal)" FORCE)
        endif()
        if(WarpX_amrex_branch)
            set(pyAMReX_amrex_branch ${WarpX_amrex_branch} CACHE STRING
                "Repository branch for WarpX_amrex_repo if(WarpX_amrex_internal)" FORCE)
        endif()
    endif()

    if(WarpX_pyamrex_internal OR WarpX_pyamrex_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        if(WarpX_pyamrex_src)
            add_subdirectory(${WarpX_pyamrex_src} _deps/localpyamrex-build/)
        else()
            FetchContent_Declare(fetchedpyamrex
                GIT_REPOSITORY ${WarpX_pyamrex_repo}
                GIT_TAG        ${WarpX_pyamrex_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedpyamrex)

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDpyamrex)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDpyamrex)
        endif()
    elseif(NOT WarpX_pyamrex_internal)
        # TODO: MPI control
        find_package(pyAMReX 25.01 CONFIG REQUIRED)
        message(STATUS "pyAMReX: Found version '${pyAMReX_VERSION}'")
    endif()
endfunction()

# local source-tree
set(WarpX_pyamrex_src ""
    CACHE PATH
    "Local path to pyAMReX source directory (preferred if set)")

# Git fetcher
option(WarpX_pyamrex_internal "Download & build pyAMReX" ON)
set(WarpX_pyamrex_repo "https://github.com/AMReX-Codes/pyamrex.git"
    CACHE STRING
    "Repository URI to pull and build pyamrex from if(WarpX_pyamrex_internal)")
set(WarpX_pyamrex_branch "6d9b9da849f5941777555ec9c9619be299d04912"
    CACHE STRING
    "Repository branch for WarpX_pyamrex_repo if(WarpX_pyamrex_internal)")

find_pyamrex()
