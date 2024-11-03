# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_3d_open_bc_poisson_solver  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_open_bc_poisson_solver  # inputs
        analysis.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()

if(WarpX_HEFFTE)
    add_warpx_test(
        test_3d_open_bc_poisson_solver_heffte # name
        3  # dims
        2  # nprocs
        inputs_test_3d_open_bc_poisson_solver_heffte  # inputs
        analysis.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()
