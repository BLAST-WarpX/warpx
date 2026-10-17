# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_current_controlled_port_picmi  # name
    2  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode 2d"  # inputs
    OFF  # analysis (the input script checks both terminal currents)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_current_controlled_port_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode 3d"  # inputs
    analysis.py  # analysis (discrete div(B) invariant)
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_EB)
    add_warpx_test(
        test_3d_current_controlled_port_eb_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_current_controlled_port_picmi.py --mode 3d --with-eb"  # inputs
        analysis.py  # analysis (discrete div(B) invariant)
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_3d_current_controlled_port_off_grid_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode 3d-off-grid"  # inputs
    analysis.py  # analysis (snapped contour, div(B), and material fields)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_current_controlled_port_hybrid_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode 3d-hybrid"  # inputs
    analysis.py  # analysis (discrete div(B) invariant)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_multiple_current_controlled_ports_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode 3d-multi"  # inputs
    analysis.py  # analysis (discrete div(B) invariant)
    OFF  # checksum
    OFF  # dependency
)

if(3 IN_LIST WarpX_DIMS AND WarpX_PYTHON)
    add_warpx_test(
        test_3d_overlapping_current_controlled_ports_rejected_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_current_controlled_port_picmi.py --mode 3d-overlap"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    set_tests_properties(
        test_3d_overlapping_current_controlled_ports_rejected_picmi.run
        PROPERTIES
            WILL_FAIL TRUE
            PASS_REGULAR_EXPRESSION "separated by more than one grid cell"
    )
endif()

add_warpx_test(
    test_3d_z_current_controlled_port_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode 3d-z"  # inputs
    analysis.py  # analysis (reversed terminals and discrete div(B) invariant)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_current_controlled_port_hybrid_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode rz-hybrid"  # inputs
    OFF  # analysis (the input script checks both terminal currents)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_radial_current_controlled_port_hybrid_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode rz-radial"  # inputs
    OFF  # analysis (the input script checks both terminal currents)
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_EB)
    add_warpx_test(
        test_rz_current_controlled_port_hybrid_eb_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_current_controlled_port_picmi.py --mode rz-hybrid --with-eb"  # inputs
        OFF  # analysis (terminal surface encloses an EB rod)
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rz_current_controlled_port_implicit_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode rz-implicit"  # inputs
    OFF  # analysis (the input script checks both terminal currents)
    OFF  # checksum
    OFF  # dependency
)

# Static refinement: per-patch material data and independent contour measurements.
if(WarpX_EB)
    foreach(source IN ITEMS port material)
        add_warpx_test(
            test_2d_current_${source}_smr_eb_picmi
            2
            1
            "inputs_test_smr_picmi.py --dims 2 --source ${source} --eb"
            "analysis_smr.py --dims 2 --source ${source} --eb"
            OFF
            OFF
        )
        add_warpx_test(
            test_3d_current_${source}_smr_eb_picmi
            3
            1
            "inputs_test_smr_picmi.py --dims 3 --source ${source} --eb"
            "analysis_smr.py --dims 3 --source ${source} --eb"
            OFF
            OFF
        )
    endforeach()
endif()

foreach(source IN ITEMS antenna port)
    add_warpx_test(
        test_2d_current_${source}_smr_picmi
        2
        1
        "inputs_test_smr_picmi.py --dims 2 --source ${source}"
        "analysis_smr.py --dims 2 --source ${source}"
        OFF
        OFF
    )
    add_warpx_test(
        test_3d_current_${source}_smr_picmi
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source ${source}"
        "analysis_smr.py --dims 3 --source ${source}"
        OFF
        OFF
    )
endforeach()

foreach(method IN ITEMS backwardeuler laxwendroff)
    add_warpx_test(
        test_3d_material_smr_${method}_picmi
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source material --method ${method}"
        "analysis_smr.py --dims 3 --source material --method ${method}"
        OFF
        OFF
    )
endforeach()

add_warpx_test(
    test_3d_current_port_smr_outside_picmi
    3
    1
    "inputs_test_smr_picmi.py --dims 3 --source port --placement outside"
    "analysis_smr.py --dims 3 --source port"
    OFF
    OFF
)

foreach(source IN ITEMS antenna port)
    add_warpx_test(
        test_3d_current_${source}_smr_picmi_restart
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source ${source} --restart ../test_3d_current_${source}_smr_picmi/diags/chk000001"
        "analysis_smr.py --dims 3 --source ${source} --reference ../test_3d_current_${source}_smr_picmi/diags/diag1000002"
        OFF
        test_3d_current_${source}_smr_picmi
    )
endforeach()

foreach(placement IN ITEMS crossing axial)
    add_warpx_test(
        test_2d_current_port_smr_${placement}_picmi
        2
        1
        "inputs_test_smr_picmi.py --dims 2 --source port --placement ${placement}"
        "analysis_smr.py --dims 2 --source port"
        OFF
        OFF
    )
    add_warpx_test(
        test_3d_current_port_smr_${placement}_picmi
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source port --placement ${placement}"
        "analysis_smr.py --dims 3 --source port"
        OFF
        OFF
    )
endforeach()


if(WarpX_EB)
    add_warpx_test(
        test_3d_current_port_smr_crossing_eb_picmi
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source port --placement crossing --eb"
        "analysis_smr.py --dims 3 --source port --eb"
        OFF
        OFF
    )
endif()

add_warpx_test(
    test_3d_current_port_smr_crossing_picmi_restart
    3
    1
    "inputs_test_smr_picmi.py --dims 3 --source port --placement crossing --restart ../test_3d_current_port_smr_crossing_picmi/diags/chk000001"
    "analysis_smr.py --dims 3 --source port --reference ../test_3d_current_port_smr_crossing_picmi/diags/diag1000002"
    OFF
    test_3d_current_port_smr_crossing_picmi
)

foreach(source IN ITEMS material port antenna)
    add_warpx_test(
        test_2d_current_${source}_smr_implicit_picmi
        2
        1
        "inputs_test_smr_picmi.py --dims 2 --source ${source} --implicit"
        "analysis_smr.py --dims 2 --source ${source} --implicit"
        OFF
        OFF
    )
    add_warpx_test(
        test_3d_current_${source}_smr_implicit_picmi
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source ${source} --implicit"
        "analysis_smr.py --dims 3 --source ${source} --implicit"
        OFF
        OFF
    )
endforeach()

foreach(placement IN ITEMS crossing axial)
    add_warpx_test(
        test_3d_current_port_smr_implicit_${placement}_picmi
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source port --placement ${placement} --implicit"
        "analysis_smr.py --dims 3 --source port --implicit"
        OFF
        OFF
    )
endforeach()

add_warpx_test(
    test_3d_current_material_smr_implicit_backward_picmi
    3
    1
    "inputs_test_smr_picmi.py --dims 3 --source material --implicit --theta 1"
    "analysis_smr.py --dims 3 --source material --implicit --theta 1"
    OFF
    OFF
)

if(WarpX_EB)
    add_warpx_test(
        test_3d_current_port_smr_implicit_eb_picmi
        3
        1
        "inputs_test_smr_picmi.py --dims 3 --source port --placement crossing --implicit --eb"
        "analysis_smr.py --dims 3 --source port --implicit --eb"
        OFF
        OFF
    )
endif()

add_warpx_test(
    test_3d_current_port_smr_implicit_picmi_restart
    3
    1
    "inputs_test_smr_picmi.py --dims 3 --source port --implicit --restart ../test_3d_current_port_smr_implicit_picmi/diags/chk000001"
    "analysis_smr.py --dims 3 --source port --implicit --reference ../test_3d_current_port_smr_implicit_picmi/diags/diag1000002"
    OFF
    test_3d_current_port_smr_implicit_picmi
)

add_warpx_test(
    test_3d_current_port_smr_implicit_plasma_picmi
    3
    1
    "inputs_test_smr_picmi.py --dims 3 --source port --implicit --plasma"
    "analysis_smr.py --dims 3 --source port --implicit"
    OFF
    OFF
)

add_warpx_test(
    test_2d_current_antenna_smr_implicit_backward_picmi
    2
    1
    "inputs_test_smr_picmi.py --dims 2 --source antenna --implicit --theta 1"
    "analysis_smr.py --dims 2 --source antenna --implicit --theta 1"
    OFF
    OFF
)

if(2 IN_LIST WarpX_DIMS AND WarpX_PYTHON)
    add_warpx_test(
        test_2d_current_port_smr_implicit_petsc_rejected_picmi
        2
        1
        "inputs_test_smr_picmi.py --dims 2 --source port --implicit --petsc"
        OFF
        OFF
        OFF
    )
    set_tests_properties(
        test_2d_current_port_smr_implicit_petsc_rejected_picmi.run
        PROPERTIES PASS_REGULAR_EXPRESSION "native AMReX GMRES"
    )
endif()

foreach(source IN ITEMS antenna port material)
    add_warpx_test(
        test_rz_current_${source}_smr_picmi
        RZ
        1
        "inputs_test_smr_picmi.py --dims RZ --source ${source}"
        "analysis_smr.py --dims RZ --source ${source}"
        OFF
        OFF
    )
endforeach()

add_warpx_test(
    test_rz_current_port_smr_picmi_restart
    RZ
    1
    "inputs_test_smr_picmi.py --dims RZ --source port --restart ../test_rz_current_port_smr_picmi/diags/chk000001"
    "analysis_smr.py --dims RZ --source port --reference ../test_rz_current_port_smr_picmi/diags/diag1000002"
    OFF
    test_rz_current_port_smr_picmi
)

foreach(source IN ITEMS antenna)
    add_warpx_test(
        test_rz_current_${source}_smr_picmi_restart
        RZ
        1
        "inputs_test_smr_picmi.py --dims RZ --source ${source} --restart ../test_rz_current_${source}_smr_picmi/diags/chk000001"
        "analysis_smr.py --dims RZ --source ${source} --reference ../test_rz_current_${source}_smr_picmi/diags/diag1000002"
        OFF
        test_rz_current_${source}_smr_picmi
    )
endforeach()

add_warpx_test(
    test_rz_current_material_smr_laxwendroff_picmi
    RZ
    1
    "inputs_test_smr_picmi.py --dims RZ --source material --method laxwendroff"
    "analysis_smr.py --dims RZ --source material --method laxwendroff"
    OFF
    OFF
)

add_warpx_test(
    test_rz_current_port_smr_annular_picmi
    RZ
    1
    "inputs_test_smr_picmi.py --dims RZ --source port --annular"
    "analysis_smr.py --dims RZ --source port"
    OFF
    OFF
)

if(WarpX_EB)
    foreach(source IN ITEMS port material)
        add_warpx_test(
            test_rz_current_${source}_smr_eb_picmi
            RZ
            1
            "inputs_test_smr_picmi.py --dims RZ --source ${source} --eb"
            "analysis_smr.py --dims RZ --source ${source} --eb"
            OFF
            OFF
        )
    endforeach()
endif()

if(RZ IN_LIST WarpX_DIMS AND WarpX_PYTHON)
    add_warpx_test(
        test_rz_current_port_smr_implicit_rejected_picmi
        RZ
        1
        "inputs_test_smr_picmi.py --dims RZ --source port --implicit"
        OFF
        OFF
        OFF
    )
    set_tests_properties(
        test_rz_current_port_smr_implicit_rejected_picmi.run
        PROPERTIES PASS_REGULAR_EXPRESSION "Cylindrical QCPML requires static explicit"
    )
endif()
