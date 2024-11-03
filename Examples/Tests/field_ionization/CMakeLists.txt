# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ionization_boost  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ionization_boost  # inputs
    analysis.py  # analysis
    diags/diag1000420  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_ionization_lab  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ionization_lab  # inputs
    analysis.py  # analysis
    diags/diag1001600  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_ionization_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ionization_picmi.py  # inputs
    analysis.py  # analysis
    diags/diag1001600  # output
    OFF  # dependency
)
