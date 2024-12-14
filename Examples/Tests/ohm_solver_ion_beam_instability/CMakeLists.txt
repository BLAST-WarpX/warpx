# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_ohm_solver_ion_beam_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_ion_beam_picmi.py --test --dim 1 --resonant"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1002500"  # checksum
    OFF  # dependency
)
label_warpx_test(test_1d_ohm_solver_ion_beam_picmi slow)
