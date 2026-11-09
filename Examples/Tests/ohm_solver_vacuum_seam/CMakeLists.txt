# Add tests (in dependency-chain order: the cell test's analysis compares
# against the edge and node run directories, so that chain must run in
# order; the guarded pair is pinned by checksums and runs independently)
#

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_edge  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_ohm_solver_vacuum_seam_edge  # inputs
    "analysis_seam.py ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_node  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_ohm_solver_vacuum_seam_node  # inputs
    "analysis_seam.py ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_vacuum_seam_edge  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_cell  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_ohm_solver_vacuum_seam_cell  # inputs
    "analysis_seam.py ../test_2d_ohm_solver_vacuum_seam_edge ../test_2d_ohm_solver_vacuum_seam_node ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_vacuum_seam_node  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_guarded_edge  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_ohm_solver_vacuum_seam_guarded_edge  # inputs
    "analysis_seam.py ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_guarded_cell  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_ohm_solver_vacuum_seam_guarded_cell  # inputs
    "analysis_seam.py ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
