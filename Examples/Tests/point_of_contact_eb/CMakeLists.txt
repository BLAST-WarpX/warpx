# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_point_of_contact_eb  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_point_of_contact_eb  # inputs
        analysis.py  # analysis
        diags/diag1/  # output
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_point_of_contact_eb  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_point_of_contact_eb  # inputs
        analysis.py  # analysis
        diags/diag1/  # output
        OFF  # dependency
    )
endif()
