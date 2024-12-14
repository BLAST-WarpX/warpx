# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_particles_in_pml  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_particles_in_pml  # inputs
    "analysis_particles_in_pml.py diags/diag1000180"  # analysis
    "analysis_default_regression.py --path diags/diag1000180"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_particles_in_pml_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_particles_in_pml_mr  # inputs
    "analysis_particles_in_pml.py diags/diag1000300"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_particles_in_pml  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particles_in_pml  # inputs
    "analysis_particles_in_pml.py diags/diag1000120"  # analysis
    "analysis_default_regression.py --path diags/diag1000120"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_particles_in_pml_mr  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particles_in_pml_mr  # inputs
    "analysis_particles_in_pml.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)
label_warpx_test(test_3d_particles_in_pml_mr slow)
