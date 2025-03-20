# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_rz_embedded_boundary_diffraction  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_embedded_boundary_diffraction  # inputs
        "analysis_fields.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()
