# spectral_precision: validates FFT, PSATD, and Hankel transforms to high
# precision across all compute backends (CPU/FFTW, CUDA, HIP, SYCL/MKL).
#
# Tier 1: standalone C++ FFT round-trip test (no WarpX binary)
# Tier 2: PICMI-based PSATD vacuum eigenmode tests (2D, 3D, RZ)

if(NOT WarpX_FFT)
    return()
endif()

# --- Tier 1: Standalone FFT round-trip test ---
# AnyFFT wrappers are compiled into ablastr; prefer 3D so all round-trip
# tests (1D/2D/3D) can run.  Fall back to whichever dim is available.
set(_FFT_SD "")
foreach(_try 3 2 RZ 1)
    if(_try IN_LIST WarpX_DIMS)
        warpx_set_suffix_dims(_FFT_SD ${_try})
        break()
    endif()
endforeach()

if(_FFT_SD AND TARGET ablastr_${_FFT_SD})
    add_executable(fft_roundtrip fft_roundtrip.cpp)
    if(WarpX_COMPUTE STREQUAL CUDA)
        set_source_files_properties(fft_roundtrip.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    target_link_libraries(fft_roundtrip PRIVATE ablastr_${_FFT_SD})
    target_include_directories(fft_roundtrip PRIVATE ${WarpX_SOURCE_DIR}/Source)

    set_target_properties(fft_roundtrip PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )

    file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fft_roundtrip_workdir)

    configure_mpiexec(1)
    add_test(
        NAME test_fft_roundtrip.run
        COMMAND ${MPI_TEST_EXE} $<TARGET_FILE:fft_roundtrip> amrex.verbose=0
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fft_roundtrip_workdir
    )
    set_property(TEST test_fft_roundtrip.run APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

# --- Tier 2: PICMI PSATD vacuum eigenmode tests ---
# Single-box variants (nprocs=1): global FFT
# Multi-box variants (nprocs=4): local FFTs with guard cell exchange

if(WarpX_PYTHON)
    # 2D single-box
    if(2 IN_LIST WarpX_DIMS)
        add_warpx_test(
            test_2d_spectral_precision  # name
            2    # dims
            1    # nprocs
            inputs_base_2d_spectral_precision.py  # inputs
            "analysis_spectral_precision.py --dim 2 --diag-dir-init diags/diag1000000 --diag-dir-final diags/diag1000040"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()

    # 2D multi-box (4 ranks, domain decomposed)
    if(2 IN_LIST WarpX_DIMS)
        add_warpx_test(
            test_2d_spectral_precision_multi  # name
            2    # dims
            4    # nprocs
            "inputs_base_2d_spectral_precision.py --ncells 64 --max_grid_size 32"  # inputs
            "analysis_spectral_precision.py --dim 2 --diag-dir-init diags/diag1000000 --diag-dir-final diags/diag1000040"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()

    # 3D single-box
    if(3 IN_LIST WarpX_DIMS)
        add_warpx_test(
            test_3d_spectral_precision  # name
            3    # dims
            1    # nprocs
            inputs_base_3d_spectral_precision.py  # inputs
            "analysis_spectral_precision.py --dim 3 --diag-dir-init diags/diag1000000 --diag-dir-final diags/diag1000040"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()

    # 3D multi-box skipped in CI: default nox=16 requires >16 valid cells per
    # box, incompatible with 32^3 grid split across 2 ranks.  Multi-box
    # decomposition is validated by 2D and RZ variants.  For manual runs with
    # more ranks, use e.g.: --ncells 64 --max_grid_size 32  (gives 8 boxes).

    # RZ z-mode precision (single-box)
    if(RZ IN_LIST WarpX_DIMS)
        add_warpx_test(
            test_rz_spectral_precision  # name
            RZ   # dims
            1    # nprocs
            inputs_base_rz_spectral_precision.py  # inputs
            "analysis_spectral_precision.py --dim rz --diag-dir-init diags/diag1000000 --diag-dir-final diags/diag1000040"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()

    # RZ multi-box (4 ranks)
    if(RZ IN_LIST WarpX_DIMS)
        add_warpx_test(
            test_rz_spectral_precision_multi  # name
            RZ   # dims
            4    # nprocs
            "inputs_base_rz_spectral_precision.py --nz 128 --max_grid_size 32"  # inputs
            "analysis_spectral_precision.py --dim rz --diag-dir-init diags/diag1000000 --diag-dir-final diags/diag1000040"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()

    # RZ Hankel eigenmode precision (single-box and multi-box).
    # Initializes the full TE eigenmode (Bz + Br satisfying div B = 0) via
    # a Python callback (scipy j0/j1) at Yee cell centres, bypassing the
    # AMReX parser.  The analysis compares against the analytical
    # standing-wave solution in physical space.
    # Tolerance: 1e-6 on CPU (~2e-7 observed, limited by DHT amplification
    # of per-step roundtrip noise); 1e-5 on GPU.
    if(WarpX_COMPUTE STREQUAL "NOACC" OR WarpX_COMPUTE STREQUAL "OMP")
        set(_eigenmode_tol "1e-6")
    else()
        set(_eigenmode_tol "1e-5")
    endif()

    if(RZ IN_LIST WarpX_DIMS)
        add_warpx_test(
            test_rz_psatd_eigenmode  # name
            RZ   # dims
            1    # nprocs
            inputs_base_rz_psatd_eigenmode.py  # inputs
            "analysis_spectral_precision.py --dim rz_eigenmode --diag-dir-init diags/diag1000000 --diag-dir-final diags/diag1000040 --tolerance ${_eigenmode_tol}"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()

    # RZ Hankel eigenmode precision (multi-box, 4 ranks)
    # Same as single-box but exercises inter-box guard cell exchange.
    if(RZ IN_LIST WarpX_DIMS)
        add_warpx_test(
            test_rz_psatd_eigenmode_multi  # name
            RZ   # dims
            4    # nprocs
            "inputs_base_rz_psatd_eigenmode.py --nz 128 --max_grid_size 32"  # inputs
            "analysis_spectral_precision.py --dim rz_eigenmode --diag-dir-init diags/diag1000000 --diag-dir-final diags/diag1000040 --tolerance ${_eigenmode_tol}"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()
endif()
