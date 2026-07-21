# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_checkpoint_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ohm_solver_checkpoint_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_checkpoint_picmi_restart  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_checkpoint_picmi.py amr.restart='../test_2d_ohm_solver_checkpoint_picmi/diags/chk000005'"  # inputs
    "analysis_default_restart.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_checkpoint_picmi  # dependency
)
