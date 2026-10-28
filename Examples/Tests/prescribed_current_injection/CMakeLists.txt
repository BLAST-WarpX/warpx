# Add tests (alphabetical order) ##############################################
#

set(prescribed_current_waveform
    "warpx.current_injection.file=${CMAKE_CURRENT_SOURCE_DIR}/current_profile.txt"
)
set(prescribed_current_override
    "warpx.current_injection.pair_1.file=${CMAKE_CURRENT_SOURCE_DIR}/current_profile_override.txt"
)

add_warpx_test(
    test_2d_prescribed_current_injection  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_prescribed_current_injection ${prescribed_current_waveform} ${prescribed_current_override}"  # inputs
    "analysis.py --jx-integral 750 --jy-integral 1500 --jz-integral -375"  # analysis
    OFF  # checksum (quantitative current-deposition oracle)
    OFF  # dependency
)

add_warpx_test(
    test_3d_prescribed_current_injection  # name
    3  # dims
    1  # nprocs
    "inputs_base_3d_prescribed_current_injection ${prescribed_current_waveform} ${prescribed_current_override}"  # inputs
    "analysis.py --jx-integral 1250 --jy-integral -625 --check-continuity"  # analysis
    OFF  # checksum (quantitative current-deposition oracle)
    OFF  # dependency
)

add_warpx_test(
    test_3d_prescribed_current_injection_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_prescribed_current_injection_picmi.py  # inputs
    "analysis.py --jx-integral 750"  # analysis
    OFF  # checksum (quantitative current-deposition and PICMI oracle)
    OFF  # dependency
)

if(3 IN_LIST WarpX_DIMS AND WarpX_APP)
    add_warpx_test(
        test_3d_prescribed_current_antenna_rejected_hybrid  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_prescribed_current_antenna_rejected_hybrid ${prescribed_current_waveform}"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    set_tests_properties(
        test_3d_prescribed_current_antenna_rejected_hybrid.run
        PROPERTIES
            WILL_FAIL TRUE
            PASS_REGULAR_EXPRESSION "impressed-current antenna.*not supported by the Hybrid-PIC solver"
    )
endif()

add_warpx_test(
    test_3d_prescribed_current_injection_restart  # name
    3  # dims
    1  # nprocs
    "inputs_base_3d_prescribed_current_injection amr.restart='../test_3d_prescribed_current_injection/diags/chk000001' ${prescribed_current_waveform} ${prescribed_current_override}"  # inputs
    "analysis.py --jx-integral 1250 --jy-integral -625"  # analysis
    OFF  # checksum (quantitative restart oracle)
    test_3d_prescribed_current_injection  # dependency
)

add_warpx_test(
    test_rz_prescribed_current_injection  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_prescribed_current_injection ${prescribed_current_waveform}"  # inputs
    "analysis.py --jz -954.929658551372 --rz-current 1500"  # analysis
    OFF  # checksum (RZ current normalization oracle)
    OFF  # dependency
)
