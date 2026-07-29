# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_resistive_drag_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_resistive_drag_picmi.py --test --eta-scale 100"  # inputs
    "analysis_resistive_drag.py --eta-scale 100"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)
