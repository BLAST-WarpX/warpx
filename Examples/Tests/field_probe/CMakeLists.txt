# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_field_probe  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_field_probe  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000544"  # checksum
        OFF  # dependency
    )
endif()
