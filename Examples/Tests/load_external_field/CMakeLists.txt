# Add tests (alphabetical order) ##############################################
#

foreach(ratio IN ITEMS "3 1" "1 3" "3 2" "2 2" "1 1")
    string(REPLACE " " "x" layout "${ratio}")
    add_warpx_test(
        test_2d_coarsened_field_samples_${layout}
        2
        1
        "inputs_base_2d_coarsened_field_samples 'diag1.coarsening_ratio=${ratio}'"
        "analysis_coarsened_field_samples.py diags/diag1000000"
        OFF
        OFF
    )
endforeach()

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
    "analysis_default_restart.py --path diags/diag1000300"  # analysis
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
    "analysis_default_restart.py --path diags/diag1000300"  # analysis
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
    "analysis_default_restart.py --path diags/diag1000300"    # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    test_3d_load_external_field_particle_time          # dependency
)

# Multi-field time dependency tests

add_warpx_test(
  test_3d_load_external_field_particle_multi_time     # name
  3                                                   # dims
  1                                                   # nprocs
  inputs_test_3d_load_external_field_particle_multi_time  # inputs
  "analysis_time_scaling.py --pf0 diags/diag1000000 --pfN diags/diag1000300 --component Bz --expected_ratio 0.0 --rtol 1e-12 --min_abs 1e-12"  # analysis
  "analysis_default_regression.py --path diags/diag1000300"  # checksum
  OFF
)

add_warpx_test(
  test_3d_load_external_field_particle_multi_time_picmi   # name
  3                                                       # dims
  1                                                       # nprocs
  inputs_test_3d_load_external_field_particle_multi_time_picmi.py  # inputs
  "analysis_time_scaling.py --pf0 diags/diag1000000 --pfN diags/diag1000300 --component Bz --expected_ratio 0.0 --rtol 1e-12 --min_abs 1e-12"  # analysis
  "analysis_default_regression.py --path diags/diag1000300"  # checksum
  OFF
)
