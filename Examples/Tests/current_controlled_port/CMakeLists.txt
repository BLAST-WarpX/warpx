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

add_warpx_test(
    test_rz_current_controlled_port_hybrid_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_current_controlled_port_picmi.py --mode rz-hybrid"  # inputs
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
