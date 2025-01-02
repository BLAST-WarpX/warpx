# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_magnetostatic_eb  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_magnetostatic_eb  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000001"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_magnetostatic_eb_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_magnetostatic_eb_picmi.py  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000001"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_magnetostatic_eb_picmi  # name
        RZ  # dims
        1   # nprocs
        inputs_test_rz_magnetostatic_eb_picmi.py  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --skip-particles"  # checksum
        OFF  # dependency
    )
endif()
