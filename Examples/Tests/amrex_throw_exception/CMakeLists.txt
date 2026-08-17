# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_amrex_throw_exception_picmi  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_amrex_throw_exception_picmi.py  # inputs
    OFF  # analysis (assertions are embedded in the input script)
    OFF  # checksum
    OFF  # dependency
)
