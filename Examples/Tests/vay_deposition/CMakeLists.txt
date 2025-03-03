# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_2d_vay_deposition  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_vay_deposition  # inputs
        "analysis.py diags/diag1000050"  # analysis
        "analysis_default_regression.py --path diags/diag1000050"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_vay_deposition  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_vay_deposition  # inputs
        "analysis.py diags/diag1000025"  # analysis
        "analysis_default_regression.py --path diags/diag1000025"  # checksum
        OFF  # dependency
    )
endif()
