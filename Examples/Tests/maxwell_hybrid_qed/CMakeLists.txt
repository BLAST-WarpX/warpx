# Add tests (alphabetical order) ##############################################
#

if(WarpX_FFT)
    add_warpx_test(
        test_2d_maxwell_hybrid_qed_solver  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_maxwell_hybrid_qed_solver  # inputs
        "analysis.py diags/diag1000300"  # analysis
        "analysis_default_regression.py --path diags/diag1000300"  # checksum
        OFF  # dependency
    )
endif()
