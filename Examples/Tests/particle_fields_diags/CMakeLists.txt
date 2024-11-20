# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_particle_fields_diags  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particle_fields_diags  # inputs
    analysis_particle_diags.py  # analysis
    diags/diag1000200  # output
    OFF  # dependency
)

# FIXME
#add_warpx_test(
#    test_3d_particle_fields_diags_single_precision  # name
#    3  # dims
#    2  # nprocs
##    inputs_test_3d_particle_fields_diags  # inputs
#    analysis_particle_diags_single.py  # analysis
#    diags/diag1000200  # output
#    OFF  # dependency
#)
