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
