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

# Insulating EB wall (boundary.eb_type = insulating) with the electron energy
# equation. Entropy-drift tolerances are set from measured values with margin:
# at rest the open-set drift is a saturating step-1 transient (-1.1e-4 at both
# 16 and 48 steps; the un-folded baseline leaks -7.3e-2 per 16 steps), under
# the compression drive it is the marker/ion transport-mismatch class
# (-1.5e-2 per 48 steps; un-folded baseline -1.9e-1).
add_warpx_test(
    test_2d_ohm_solver_insulating_eb_at_rest_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_insulating_eb_picmi.py --case at-rest --test"  # inputs
    "analysis_insulating_eb.py --tol-entropy 5.0e-4"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_insulating_eb_compress_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_insulating_eb_picmi.py --case compress --test"  # inputs
    "analysis_insulating_eb.py --tol-entropy 5.0e-2"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)
