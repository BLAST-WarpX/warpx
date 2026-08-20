# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_electron_energy_adiabat_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case adiabat --test"  # inputs
    "analysis_adiabat.py --tol-median 0.01 --tol-max 0.05"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_joule_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case joule --test --eta-scale 100"  # inputs
    "analysis_joule.py --eta-scale 100"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_qei_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case qei --test"  # inputs
    "analysis_qei.py --nu-ei 1e6"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_vacuum_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case vacuum --test"  # inputs
    "analysis_vacuum.py"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)
