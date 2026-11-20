# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_dielectric_slab_capacitor_picmi  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_dielectric_slab_capacitor_picmi.py  # inputs
        analysis.py  # analysis
        "analysis_default_regression.py --path diags/diag1"  # checksum
        OFF  # dependency
    )
endif()
