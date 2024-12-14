# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000001"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_mixed_bc  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_mixed_bc  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000001"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_picmi  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_picmi.py  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000002"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb  # inputs
        "analysis_rz.py diags/diag1000001"  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --skip-particles"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb_mr  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb_mr  # inputs
        "analysis_rz_mr.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()
