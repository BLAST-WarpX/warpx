# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_3d_open_bc_poisson_solver  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_open_bc_poisson_solver  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --rtol 1e-2"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_open_bc_poisson_solver_sliced  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_open_bc_poisson_solver_sliced  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --rtol 1e-2"  # checksum
        OFF  # dependency
    )
endif()
