# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_load_external_field_grid_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_load_external_field_grid_picmi.py  # inputs
    "analysis_3d.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_load_external_field_particle_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_load_external_field_particle_picmi.py  # inputs
    "analysis_3d.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_load_external_field_grid  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_load_external_field_grid  # inputs
    "analysis_rz.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_load_external_field_grid_restart  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_load_external_field_grid_restart  # inputs
    "analysis_default_restart.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    test_rz_load_external_field_grid  # dependency
)

add_warpx_test(
    test_rz_load_external_field_particles  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_load_external_field_particles  # inputs
    "analysis_rz.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_load_external_field_particles_restart  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_load_external_field_particles_restart  # inputs
    "analysis_default_restart.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    test_rz_load_external_field_particles  # dependency
)


# ---- Time dependency tests ----

add_warpx_test(
    test_3d_load_external_field_particle_time_picmi   # name
    3                                                 # dims
    1                                                 # nprocs
    inputs_test_3d_load_external_field_particle_time_picmi.py   # inputs
    "analysis_time_scaling.py --pf0 diags/diag1000000 --pfN diags/diag1000300 --component Bz --expected_ratio 0.5 --rtol 1e-6 --min_abs 1e-12"
    "analysis_default_regression.py --path diags/diag1000300"   # checksum (optional regression)
    OFF                                              # dependency
)

add_warpx_test(
    test_3d_load_external_field_particle_time         # name
    3                                                 # dims
    1                                                 # nprocs
    inputs_test_3d_load_external_field_particle_time  # inputs
    "analysis_time_scaling.py --pf0 diags/diag1000000 --pfN diags/diag1000300 --component Bz --expected_ratio 0.5 --rtol 1e-6 --min_abs 1e-12"
     "analysis_default_regression.py --path diags/diag1000300"   # checksum
    OFF
)

add_warpx_test(
    test_3d_load_external_field_particle_time_restart  # name
    3                                                  # dims
    1                                                  # nprocs
    inputs_test_3d_load_external_field_particle_time_restart  # inputs
    "analysis_default_restart.py diags/diag1000300"    # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    test_3d_load_external_field_particle_time          # dependency
)
