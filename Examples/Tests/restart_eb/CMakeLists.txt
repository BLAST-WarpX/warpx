# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_eb_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_eb_picmi.py  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000060"  # checksum
        OFF  # dependency
    )
endif()

# FIXME
#if(WarpX_EB)
#    add_warpx_test(
#        test_3d_eb_picmi_restart  # name
#        3  # dims
#        1  # nprocs
#        "inputs_test_3d_eb_picmi.py amr.restart='../test_3d_eb_picmi/diags/chk000030'"  # inputs
#        "analysis_default_restart.py diags/diag1000060"  # analysis
#        "analysis_default_regression.py --path diags/diag1000060"  # checksum
#        test_3d_eb_picmi  # dependency
#    )
#endif()
