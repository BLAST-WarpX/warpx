# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ohm_solver_cylinder_compression_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_cylinder_compression_picmi.py --test"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010 --rtol 5e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_3d_ohm_solver_cylinder_compression_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_cylinder_compression_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_cylinder_compression_picmi.py --test"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020 --rtol 1e-6"  # output
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_cylinder_compression_picmi slow)
