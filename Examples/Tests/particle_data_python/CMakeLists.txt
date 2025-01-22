# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_particle_attr_access_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_particle_attr_access_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_particle_attr_access_unique_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_particle_attr_access_picmi.py --unique"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_prev_positions_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_prev_positions_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
