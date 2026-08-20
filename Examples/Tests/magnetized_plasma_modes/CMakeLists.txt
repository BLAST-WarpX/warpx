# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_darwin_solver_em_modes_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_em_modes_picmi.py --test --dim 1 --bdir z --darwin"  # inputs
    "analysis.py --analyze_darwin_sim"  # analysis
    "analysis_default_regression.py --path diags/field_diag000050"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_ohm_solver_em_modes_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_em_modes_picmi.py --test --dim 1 --bdir z --ohm"  # inputs
    "analysis.py --analyze_ohm_sim"  # analysis
    "analysis_default_regression.py --path diags/field_diag000250"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_ohm_solver_em_modes_rkf45_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_em_modes_picmi.py --test --dim 1 --bdir z --ohm --use_rkf45"  # inputs
    "analysis.py --analyze_ohm_sim"  # analysis
    "analysis_default_regression.py --path diags/field_diag000250"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_darwin_solver_em_modes_es_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_em_modes_picmi.py --test --dim 2 --bdir z --darwin --include_es_solver"  # inputs
    "analysis.py --analyze_darwin_sim"  # analysis
    "analysis_default_regression.py --path diags/field_diag000050"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_em_modes_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_picmi.py --test"  # inputs
    "analysis_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100 --rtol 1e-6"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_picmi slow)
