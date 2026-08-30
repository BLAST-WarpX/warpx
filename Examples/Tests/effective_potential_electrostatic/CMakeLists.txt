# Add tests (alphabetical order) ##############################################
#

if(WarpX_PRECISION STREQUAL "DOUBLE")
    set(effective_potential_checksum
        "analysis_default_regression.py --path diags/field_diag/")
else()
    set(effective_potential_checksum OFF)
endif()

add_warpx_test(
    test_3d_effective_potential_electrostatic_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_effective_potential_electrostatic_picmi.py --precision ${WarpX_PRECISION}"  # inputs
    "analysis.py --precision ${WarpX_PRECISION}"  # analysis
    "${effective_potential_checksum}"  # checksum
    OFF  # dependency
)
