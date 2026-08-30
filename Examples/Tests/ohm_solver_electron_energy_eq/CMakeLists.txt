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
    test_2d_ohm_solver_electron_energy_latent_adiabat_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case latent_adiabat --test"  # inputs
    "analysis_latent_adiabat.py"  # analysis
    OFF  # checksum (the nonlinear adiabat is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_latent_joule_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case latent_joule --test --eta-scale 100"  # inputs
    "analysis_joule.py --eta-scale 100 --latent-energy-eV 2000 --transition-temperature-eV 500 --latent-sharpness 6 --out latent_joule_check.png"  # analysis
    OFF  # checksum (the nonlinear caloric source is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_latent_qei_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case latent_qei --test"  # inputs
    "analysis_latent_qei.py --nu-ei 1e6 --latent-energy-eV 1000 --transition-temperature-eV 200 --latent-sharpness 4"  # analysis
    OFF  # checksum (the nonlinear caloric exchange is checked analytically)
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
