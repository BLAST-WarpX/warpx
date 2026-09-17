# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_particle_sink_picmi  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_particle_sink_picmi.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
