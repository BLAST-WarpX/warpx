# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_electrostatic_sphere  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_electrostatic_sphere  # inputs
    "analysis_electrostatic_sphere.py diags/diag1000030"  # analysis
    "analysis_default_regression.py --path diags/diag1000030"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_electrostatic_sphere_lab_frame  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_electrostatic_sphere_lab_frame  # inputs
    "analysis_electrostatic_sphere.py diags/diag1000030"  # analysis
    "analysis_default_regression.py --path diags/diag1000030"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_electrostatic_sphere_lab_frame_mr_emass_10  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_electrostatic_sphere_lab_frame_mr_emass_10  # inputs
    "analysis_electrostatic_sphere.py diags/diag1000002"  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_electrostatic_sphere_rel_nodal  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_electrostatic_sphere_rel_nodal  # inputs
    "analysis_electrostatic_sphere.py diags/diag1000030"  # analysis
    "analysis_default_regression.py --path diags/diag1000030"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_electrostatic_sphere_adaptive  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_electrostatic_sphere_adaptive  # inputs
    "analysis_electrostatic_sphere.py diags/diag1000054"  # analysis
    "analysis_default_regression.py --path diags/diag1000054"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_electrostatic_sphere  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_electrostatic_sphere  # inputs
    "analysis_electrostatic_sphere.py diags/diag1000030"  # analysis
    "analysis_default_regression.py --path diags/diag1000030"  # checksum
    OFF  # dependency
)
