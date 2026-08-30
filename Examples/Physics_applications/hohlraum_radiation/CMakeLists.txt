# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_hohlraum_radiation_hybrid  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_hohlraum_radiation_hybrid  # inputs
    "analysis.py --coupling hybrid"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_hohlraum_radiation_hybrid_multigroup  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_hohlraum_radiation_hybrid_multigroup  # inputs
    "analysis.py --coupling hybrid --multigroup"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_hohlraum_radiation_kinetic  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_hohlraum_radiation_kinetic  # inputs
    "analysis.py --coupling kinetic"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_hohlraum_radiation_kinetic_multigroup  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_hohlraum_radiation_kinetic_multigroup  # inputs
    "analysis.py --coupling kinetic --multigroup"  # analysis
    OFF  # checksum
    OFF  # dependency
)
