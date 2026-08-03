# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_electrostatic_macroscopic_solver_picmi  # name
    1  # dims
    1  # nprocs
    inputs_test_2d_electrostatic_macroscopic_solver_picmi.py  # inputs
    analysis.py  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)
