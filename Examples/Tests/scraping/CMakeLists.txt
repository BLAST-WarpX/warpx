# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_rz_scraping  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_scraping  # inputs
        "analysis_rz.py diags/diag1000037"  # analysis
        "analysis_default_regression.py --path diags/diag1000037"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_scraping_filter  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_scraping_filter  # inputs
        "analysis_rz_filter.py diags/diag1000037"  # analysis
        "analysis_default_regression.py --path diags/diag1000037"  # checksum
        OFF  # dependency
    )
endif()
