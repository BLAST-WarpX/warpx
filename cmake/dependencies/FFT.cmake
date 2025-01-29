if(ABLASTR_FFT)
    # Helper Functions ############################################################
    #
    option(WarpX_FFTW_IGNORE_OMP "Ignore FFTW3 OpenMP support, even if found" OFF)
    mark_as_advanced(WarpX_FFTW_IGNORE_OMP)

    # Set the WarpX_FFTW_OMP=1 define on WarpX::thirdparty::FFT if TRUE and print
    # a message
    #
    function(fftw_add_define HAS_FFTW_OMP_LIB)
        if(HAS_FFTW_OMP_LIB)
            message(STATUS "FFTW: Found OpenMP support")
            target_compile_definitions(WarpX::thirdparty::FFT INTERFACE WarpX_FFTW_OMP=1)
        else()
            message(STATUS "FFTW: Could NOT find OpenMP support")
        endif()
    endfunction()

    # Check if the found FFTW install location has an _omp library, e.g.,
    # libfftw3(f)_omp.(a|so) shipped and if yes, set the WarpX_FFTW_OMP=1 define.
    #
    function(fftw_check_omp library_paths fftw_precision_suffix)
        find_library(HAS_FFTW_OMP_LIB fftw3${fftw_precision_suffix}_omp
            PATHS ${library_paths}
            # this is intentional, so we don't mix different FFTW installs
            # and only check what is in the location hinted by the
            # "library_paths" variable
            NO_DEFAULT_PATH
            NO_PACKAGE_ROOT_PATH
            NO_CMAKE_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH
        )
        if(HAS_FFTW_OMP_LIB)
            # the .pc files here forget to link the _omp.a/so files
            # explicitly - we add those manually to avoid any trouble,
            # e.g., in static builds.
            target_link_libraries(WarpX::thirdparty::FFT INTERFACE ${HAS_FFTW_OMP_LIB})
        endif()

        fftw_add_define("${HAS_FFTW_OMP_LIB}")
    endfunction()


    # Various FFT implementations that we want to use #############################
    #

    # cuFFT  (CUDA)
    if(WarpX_COMPUTE STREQUAL CUDA)
        # nothing to do (cuFFT is part of the CUDA SDK)
        # TODO: check if `find_package` search works for cuFFT

    # rocFFT (HIP)
    elseif(WarpX_COMPUTE STREQUAL HIP)
        find_package(rocfft REQUIRED)

    elseif(WarpX_COMPUTE STREQUAL SYCL)
        # nothing to do (oneMKL is part of oneAPI)
        # TODO: check if `find_package` search works for oneMKL

    # FFTW   (NOACC, OMP)
    else()
        # On Windows, try searching for FFTW3(f)Config.cmake files first
        #   Installed .pc files wrongly and unconditionally add -lm
        #   https://github.com/FFTW/fftw3/issues/236

        # On Linux & macOS, note Autotools install bug:
        #   https://github.com/FFTW/fftw3/issues/235
        # Thus, rely on .pc files

        set(WarpX_FFTW_SEARCH_VALUES PKGCONFIG CMAKE)
        set(WarpX_FFTW_SEARCH_DEFAULT PKGCONFIG)
        if(WIN32)
            set(WarpX_FFTW_SEARCH_DEFAULT CMAKE)
        endif()
        set(WarpX_FFTW_SEARCH ${WarpX_FFTW_SEARCH_DEFAULT}
            CACHE STRING "FFTW search method (PKGCONFIG/CMAKE)")
        set_property(CACHE WarpX_FFTW_SEARCH PROPERTY STRINGS ${WarpX_FFTW_SEARCH_VALUES})
        if(NOT WarpX_FFTW_SEARCH IN_LIST WarpX_FFTW_SEARCH_VALUES)
            message(FATAL_ERROR "WarpX_FFTW_SEARCH (${WarpX_FFTW_SEARCH}) must be one of ${WarpX_FFTW_SEARCH_VALUES}")
        endif()
        mark_as_advanced(WarpX_FFTW_SEARCH)

        # floating point precision suffixes: float, double and quad precision
        if(WarpX_PRECISION STREQUAL "DOUBLE")
            set(HFFTWp "")
        else()
            set(HFFTWp "f")
        endif()

        if(WarpX_FFTW_SEARCH STREQUAL CMAKE)
            find_package(FFTW3${HFFTWp} CONFIG REQUIRED)
            set(WarpX_FFTW_LIBRARY_DIRS "${FFTW3${HFFTWp}_LIBRARY_DIRS}")
            message(STATUS "Found FFTW: ${FFTW3${HFFTWp}_DIR} (found version \"${FFTW3${HFFTWp}_VERSION}\")")
        else()
            find_package(PkgConfig REQUIRED QUIET)
            pkg_check_modules(fftw3${HFFTWp} REQUIRED IMPORTED_TARGET fftw3${HFFTWp})
            message(STATUS "Found FFTW: ${fftw3${HFFTWp}_PREFIX}")
            if(fftw3${HFFTWp}_LIBRARY_DIRS)
                set(WarpX_FFTW_LIBRARY_DIRS "${fftw3${HFFTWp}_LIBRARY_DIRS}")
            else()
                set(WarpX_FFTW_LIBRARY_DIRS "${fftw3${HFFTWp}_LIBDIR}")
            endif()
        endif()
    endif()

    # create an IMPORTED target: WarpX::thirdparty::FFT
    if(WarpX_COMPUTE STREQUAL CUDA)
        # CUDA_ADD_CUFFT_TO_TARGET(WarpX::thirdparty::FFT)
        warpx_make_third_party_includes_system(cufft FFT)
    elseif(WarpX_COMPUTE STREQUAL HIP)
        warpx_make_third_party_includes_system(roc::rocfft FFT)
    elseif(WarpX_COMPUTE STREQUAL SYCL)
        warpx_make_third_party_includes_system(AMReX::SYCL FFT)
    else()
        if(WarpX_FFTW_SEARCH STREQUAL CMAKE)
            warpx_make_third_party_includes_system(FFTW3::fftw3${HFFTWp} FFT)
        else()
            warpx_make_third_party_includes_system(PkgConfig::fftw3${HFFTWp} FFT)
        endif()
        if(WarpX_COMPUTE STREQUAL OMP)
            if(WarpX_FFTW_IGNORE_OMP)
                message(STATUS "FFTW: Requested to IGNORE OpenMP support")
            else()
                fftw_check_omp("${WarpX_FFTW_LIBRARY_DIRS}" "${HFFTWp}")
            endif()
        else()
            message(STATUS "FFTW: Did NOT search for OpenMP support (WarpX_COMPUTE!=OMP)")
        endif()
    endif()
endif(ABLASTR_FFT)
