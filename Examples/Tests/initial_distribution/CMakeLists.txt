# Add tests (alphabetical order) ##############################################
#

foreach(power IN ITEMS 0 1 2)
    add_warpx_test(
        test_rz_initial_radial_profile_power${power}
        RZ
        1
        "inputs_base_rz_initial_radial_profile my_constants.power=${power}"
        "analysis_radial_profile.py --power ${power} --precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
endforeach()

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
