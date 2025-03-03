# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_em_particle_absorption_sh_factor_1  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_em_particle_absorption_sh_factor_1  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_em_particle_absorption_sh_factor_2  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_em_particle_absorption_sh_factor_2  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_embedded_boundary_em_particle_absorption_sh_factor_3  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_embedded_boundary_em_particle_absorption_sh_factor_3  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_em_particle_absorption_sh_factor_1  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_embedded_boundary_em_particle_absorption_sh_factor_1  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_em_particle_absorption_sh_factor_2  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_embedded_boundary_em_particle_absorption_sh_factor_2  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_embedded_boundary_em_particle_absorption_sh_factor_3  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_embedded_boundary_em_particle_absorption_sh_factor_3  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_embedded_boundary_em_particle_absorption_sh_factor_1  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_embedded_boundary_em_particle_absorption_sh_factor_1  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_embedded_boundary_em_particle_absorption_sh_factor_2  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_embedded_boundary_em_particle_absorption_sh_factor_2  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_embedded_boundary_em_particle_absorption_sh_factor_3  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_embedded_boundary_em_particle_absorption_sh_factor_3  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()
