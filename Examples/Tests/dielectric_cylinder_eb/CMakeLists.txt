# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_dielectric_cylinder_eb  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_dielectric_cylinder  # inputs
        "analysis.py diags/diag1000001"  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --skip-particles"  # checksum
        OFF  # dependency
    )
endif()
