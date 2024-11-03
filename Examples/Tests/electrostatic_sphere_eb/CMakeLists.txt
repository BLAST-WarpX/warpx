# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb  # inputs
        analysis.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_mixed_bc  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_mixed_bc  # inputs
        analysis_default_regression.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_picmi  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_picmi.py  # inputs
        analysis.py  # analysis
        diags/diag1000002  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb  # inputs
        analysis_rz.py  # analysis
        diags/diag1000001  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb_mr  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb_mr  # inputs
        analysis_rz_mr.py  # analysis
        diags/diag1/  # output
        OFF  # dependency
    )
endif()
