# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_particle_fields_diags  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particle_fields_diags  # inputs
    "analysis_particle_diags.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

# FIXME
#add_warpx_test(
#    test_3d_particle_fields_diags_single_precision  # name
#    3  # dims
#    2  # nprocs
#    inputs_test_3d_particle_fields_diags  # inputs
#    "analysis_particle_diags_single.py diags/diag1000200"  # analysis
#    "analysis_default_regression.py --path diags/diag1000200 --rtol 1e-3"  # checksum
#    OFF  # dependency
#)
