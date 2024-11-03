# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_energy_conserving_thermal_plasma  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_energy_conserving_thermal_plasma  # inputs
    analysis.py  # analysis
    diags/diag1000500  # output
    OFF  # dependency
)
