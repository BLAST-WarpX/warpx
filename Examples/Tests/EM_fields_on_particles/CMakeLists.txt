# add tests

add_warpx_test(
    test_2d_particle_EM_diagnostics # name
    2 # dims
    2 # nprocs
    inputs_test_2d_particle_EM_diagnostics # inputs
    "analysis_2d.py" # analysis
    "analysis_default_regression.py --path diags/diag_checksum/" # checksum
    OFF # dependency
)
