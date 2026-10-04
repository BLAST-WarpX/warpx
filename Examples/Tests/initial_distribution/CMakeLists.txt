# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_initial_radial_profile_power0
    RZ
    1
    "inputs_base_rz_initial_radial_profile my_constants.power=0"
    "analysis_radial_profile.py --power 0 --precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_rz_initial_radial_profile_power1
    RZ
    1
    "inputs_base_rz_initial_radial_profile my_constants.power=1"
    "analysis_radial_profile.py --power 1 --precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_rz_initial_radial_profile_power2
    RZ
    1
    "inputs_base_rz_initial_radial_profile my_constants.power=2"
    "analysis_radial_profile.py --power 2 --precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_rz_initial_radial_profile_mpi
    RZ
    2
    "inputs_base_rz_initial_radial_profile my_constants.power=1"
    "analysis_radial_profile.py --power 1 --precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)

add_warpx_test(
    test_3d_initial_distribution_prepare  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_initial_distribution_prepare.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_initial_distribution  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_initial_distribution  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    test_3d_initial_distribution_prepare  # dependency
)
