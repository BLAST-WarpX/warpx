# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_magnetostatic_eb  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_magnetostatic_eb  # inputs
        analysis_default_regression.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_magnetostatic_eb_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_magnetostatic_eb_picmi.py  # inputs
        analysis_default_regression.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_magnetostatic_eb_picmi  # name
        RZ  # dims
        1   # nprocs
        inputs_test_rz_magnetostatic_eb_picmi.py  # inputs
        analysis_rz.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()
