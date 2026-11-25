# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_dielectric_surface_charge  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_dielectric_surface_charge  # inputs
        "analysis.py diags/diag1000120"  # analysis
        "analysis_default_regression.py --path diags/diag1000120"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_dielectric_surface_charge_fast_tiling  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_dielectric_surface_charge_fast_tiling  # inputs
        "analysis.py diags/diag1000024 5.5e-3"  # analysis
        "analysis_default_regression.py --path diags/diag1000024"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_dielectric_surface_charge_periodic  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_dielectric_surface_charge_periodic  # inputs
        "analysis_face_charge.py diags/diag1000012 x 1.171875e-3"  # analysis
        "analysis_default_regression.py --path diags/diag1000012"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_dielectric_surface_charge_reflecting  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_dielectric_surface_charge_reflecting  # inputs
        "analysis_face_charge.py diags/diag1000030 z 1.171875e-4"  # analysis
        "analysis_default_regression.py --path diags/diag1000030"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_dielectric_surface_charge_restart  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_dielectric_surface_charge_restart  # inputs
        "analysis_default_restart.py --path diags/diag1000120"  # analysis
        "analysis_default_regression.py --path diags/diag1000120"  # checksum
        test_2d_dielectric_surface_charge  # dependency
    )
endif()
