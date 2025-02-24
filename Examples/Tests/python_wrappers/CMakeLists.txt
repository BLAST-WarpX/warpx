# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_2d_python_wrappers_picmi  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_python_wrappers_picmi.py  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000100"  # checksum
        OFF  # dependency
    )
endif()
