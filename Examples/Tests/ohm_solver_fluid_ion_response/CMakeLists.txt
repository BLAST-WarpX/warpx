# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_fluid_ion_response_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ohm_solver_fluid_ion_response_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

# the same deck with the fluid ion response, compared with the particle-push run above
add_warpx_test(
    test_2d_ohm_solver_fluid_ion_response_fluid_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_fluid_ion_response_picmi.py --fluid"  # inputs
    "analysis_fluid_ion_response.py --ref ../test_2d_ohm_solver_fluid_ion_response_picmi"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    test_2d_ohm_solver_fluid_ion_response_picmi  # dependency
)
