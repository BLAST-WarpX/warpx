# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_particle_scrape  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_particle_scrape  # inputs
        "analysis_scrape.py diags/diag1000060"  # analysis
        "analysis_default_regression.py --path diags/diag1000060"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_particle_scrape_picmi  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_particle_scrape_picmi.py  # inputs
        "analysis_scrape.py diags/diag1000060"  # analysis
        "analysis_default_regression.py --path diags/diag1000060"  # checksum
        OFF  # dependency
    )
endif()
