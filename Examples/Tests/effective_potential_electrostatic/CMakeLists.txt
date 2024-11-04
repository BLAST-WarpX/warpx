# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_effective_potential_electrostatic_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_effective_potential_electrostatic_picmi.py  # inputs
    analysis.py  # analysis
    diags/field_diag/  # output
    OFF  # dependency
)
