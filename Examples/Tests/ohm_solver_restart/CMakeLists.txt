# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_adaptive_checkpoint  # name
    2  # dims
    1  # nprocs
    inputs_base_2d_ohm_solver_adaptive_checkpoint  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_adaptive_checkpoint_restart  # name
    2  # dims
    1  # nprocs
    "inputs_base_2d_ohm_solver_adaptive_checkpoint amr.restart='../test_2d_ohm_solver_adaptive_checkpoint/diags/chk000003' hybrid_pic_model.substeps=2"  # inputs
    "analysis_adaptive_substeps.py"  # analysis
    OFF  # checksum
    test_2d_ohm_solver_adaptive_checkpoint  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_checkpoint_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ohm_solver_checkpoint_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_checkpoint_picmi_restart  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_checkpoint_picmi.py amr.restart='../test_2d_ohm_solver_checkpoint_picmi/diags/chk000005'"  # inputs
    "analysis_default_restart.py --path diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_checkpoint_picmi  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_external_field_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ohm_solver_external_field_segmented_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_external_field_segmented_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_external_field_segmented_picmi.py --segments 3,3,4"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

# A saved adaptive count must not override an explicitly increased RK4 count.
add_warpx_test(
    test_2d_ohm_solver_fixed_substeps_restart  # name
    2  # dims
    1  # nprocs
    "inputs_base_2d_ohm_solver_adaptive_checkpoint amr.restart='../test_2d_ohm_solver_adaptive_checkpoint/diags/chk000003' hybrid_pic_model.use_rkf45=0 hybrid_pic_model.substeps=80"  # inputs
    "analysis_adaptive_substeps.py 80"  # analysis
    OFF  # checksum
    test_2d_ohm_solver_adaptive_checkpoint  # dependency
)
