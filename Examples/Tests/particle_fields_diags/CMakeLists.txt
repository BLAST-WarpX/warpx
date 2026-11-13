# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_refined_charge_density  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_refined_charge_density  # inputs
    "analysis_refined_charge_density.py diags/diag1000000 2d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_particle_fields_diags  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_particle_fields_diags  # inputs
    "analysis_particle_diags.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_refined_charge_density  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_refined_charge_density  # inputs
    "analysis_refined_charge_density.py diags/diag1000000 3d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_refined_charge_density  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_refined_charge_density  # inputs
    "analysis_refined_charge_density.py diags/diag1000000 rz"  # analysis
    OFF  # checksum
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
