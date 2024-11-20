# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_2d_vay_deposition  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_vay_deposition  # inputs
        analysis.py  # analysis
        diags/diag1000050  # output
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_vay_deposition  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_vay_deposition  # inputs
        analysis.py  # analysis
        diags/diag1000025  # output
        OFF  # dependency
    )
endif()
