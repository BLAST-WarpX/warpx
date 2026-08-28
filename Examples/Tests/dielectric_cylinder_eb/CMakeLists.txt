# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_conducting_cylinder_dielectric_shell_eb  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_conducting_cylinder_dielectric_shell_eb  # inputs
        "analysis_conducting_cylinder.py diags/diag1000001"  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --skip-particles"  # checksum
        OFF  # dependency
    )

    set(_unit_sphere_stl "${CMAKE_CURRENT_SOURCE_DIR}/unit_sphere.stl")
    set(_stl_inputs inputs_test_3d_conducting_sphere_dielectric_shell_stl)
    list(APPEND _stl_inputs "shell.stl_file=${_unit_sphere_stl}")
    list(JOIN _stl_inputs " " _stl_inputs)
    add_warpx_test(
        test_3d_conducting_sphere_dielectric_shell_stl  # name
        3  # dims
        1  # nprocs
        "${_stl_inputs}"  # inputs
        "analysis_conducting_sphere.py diags/diag1000001"  # analysis
        OFF  # checksum
        OFF  # dependency
    )

endif()
