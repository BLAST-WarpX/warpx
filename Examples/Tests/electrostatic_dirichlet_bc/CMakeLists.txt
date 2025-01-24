# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_dirichlet_bc  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_dirichlet_bc  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_dirichlet_bc_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_dirichlet_bc_picmi.py  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)
