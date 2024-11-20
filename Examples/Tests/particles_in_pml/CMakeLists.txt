# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_particles_in_pml  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_particles_in_pml  # inputs
    analysis_particles_in_pml.py  # analysis
    diags/diag1000180  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_particles_in_pml_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_particles_in_pml_mr  # inputs
    analysis_particles_in_pml.py  # analysis
    diags/diag1000300  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_particles_in_pml  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particles_in_pml  # inputs
    analysis_particles_in_pml.py  # analysis
    diags/diag1000120  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_particles_in_pml_mr  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particles_in_pml_mr  # inputs
    analysis_particles_in_pml.py  # analysis
    diags/diag1000200  # output
    OFF  # dependency
)
label_warpx_test(test_3d_particles_in_pml_mr slow)
