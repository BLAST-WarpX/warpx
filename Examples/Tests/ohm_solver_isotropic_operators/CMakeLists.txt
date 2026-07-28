# Add tests (alphabetical order) ##############################################
#

# Self-asserting unit batteries of the isotropized hybrid Ohm's-law operators
# (hyper-resistivity Laplacian, resistive corner-curl, electron-pressure
# gradient). Each suite runs its batteries as subprocess children (each
# battery needs its own solver setup, and WarpX initializes once per process).
add_warpx_test(
    test_2d_ohm_solver_isotropic_operators_picmi  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_ohm_solver_isotropic_operators_picmi.py --suite staggered"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_isotropic_operators_collocated_picmi  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_ohm_solver_isotropic_operators_picmi.py --suite collocated"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ohm_solver_isotropic_operators_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_3d_ohm_solver_isotropic_operators_picmi.py --suite staggered"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ohm_solver_isotropic_operators_collocated_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_3d_ohm_solver_isotropic_operators_picmi.py --suite collocated"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
