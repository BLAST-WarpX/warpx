# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_field_probe  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_field_probe  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000544"  # checksum
        OFF  # dependency
    )
endif()

foreach(nprocs IN ITEMS 1 2)
    foreach(refined IN ITEMS 0 1)
        if(refined)
            set(options "amr.max_level=1 warpx.fine_tag_lo=-0.25 -0.5 -0.5 warpx.fine_tag_hi=0.25 0.5 0.5 warpx.pml_ncell=4 warpx.pml_delta=4")
        else()
            set(options "")
        endif()
        add_warpx_test(
            test_3d_field_probe_refinement_${refined}_${nprocs}
            3
            ${nprocs}
            "inputs_base_3d_refinement ${options}"
            analysis_refinement.py
            OFF
            OFF
        )
    endforeach()
endforeach()
