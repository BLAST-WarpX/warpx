# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_prescribed_current_injection     # name
    3                                        # dims
    4                                        # nprocs
    "inputs_test_3d_prescribed_current_injection warpx.current_injection.file=${CMAKE_CURRENT_SOURCE_DIR}/current_profile.txt"  # inputs
    "analysis.py"                            # analysis
    OFF                                      # checksum (physics-based check only)
    OFF                                      # dependency
)
