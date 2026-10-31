# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_magnetostatic_ring_centering_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_magnetostatic_ring_centering_picmi.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
