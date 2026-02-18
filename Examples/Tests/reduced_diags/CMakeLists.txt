# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_reduced_diags  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_reduced_diags  # inputs
    "analysis_reduced_diags.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_reduced_diags_load_balance_costs_heuristic  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_reduced_diags_load_balance_costs_heuristic  # inputs
    "analysis_reduced_diags_load_balance_costs.py diags/diag1000003"  # analysis
    "analysis_default_regression.py --path diags/diag1000003"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_reduced_diags_load_balance_costs_timers  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_reduced_diags_load_balance_costs_timers  # inputs
    "analysis_reduced_diags_load_balance_costs.py diags/diag1000003"  # analysis
    "analysis_default_regression.py --path diags/diag1000003"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_reduced_diags_load_balance_costs_timers_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_reduced_diags_load_balance_costs_timers_picmi.py  # inputs
    "analysis_reduced_diags_load_balance_costs.py diags/diag1000003"  # analysis
    "analysis_default_regression.py --path diags/diag1000003"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_3d_reduced_diags_load_balance_costs_timers_psatd  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_reduced_diags_load_balance_costs_timers_psatd  # inputs
        "analysis_reduced_diags_load_balance_costs.py diags/diag1000003"  # analysis
        "analysis_default_regression.py --path diags/diag1000003"  # checksum
        OFF  # dependency
    )
endif()
