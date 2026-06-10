# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_hr_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 64"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_picmi  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_conformal_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --conformal"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_hr_picmi  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_conformal_hr_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 64 --conformal"  # inputs
        "analysis.py --stair-lo ../test_3d_ohm_solver_eb_diffusion_picmi/diags/diag1000200 --stair-hi ../test_3d_ohm_solver_eb_diffusion_hr_picmi/diags/diag1000200 --conformal-lo ../test_3d_ohm_solver_eb_diffusion_conformal_picmi/diags/diag1000200 --conformal-hi diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_conformal_picmi  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_pec_j_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --pec-j"  # inputs
        "analysis_pec_j.py --path diags/diag1000024"  # analysis
        "analysis_default_regression.py --path diags/diag1000024"  # checksum
        test_3d_ohm_solver_eb_diffusion_conformal_hr_picmi  # dependency
    )
endif()
