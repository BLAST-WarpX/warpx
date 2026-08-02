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
    "analysis.py --jx-integral 500 --jy-integral 1000 --jz-integral -250"  # analysis
    OFF  # checksum (quantitative current-deposition oracle)
    OFF  # dependency
)

add_warpx_test(
    test_3d_prescribed_current_injection  # name
    3  # dims
    1  # nprocs
    "inputs_base_3d_prescribed_current_injection ${prescribed_current_waveform} ${prescribed_current_override}"  # inputs
    "analysis.py --jx-integral 1000 --jy-integral -500"  # analysis
    OFF  # checksum (quantitative current-deposition oracle)
    OFF  # dependency
)

add_warpx_test(
    test_3d_prescribed_current_injection_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_prescribed_current_injection_picmi.py  # inputs
    "analysis.py --jx-integral 500"  # analysis
    OFF  # checksum (quantitative current-deposition and PICMI oracle)
    OFF  # dependency
)

add_warpx_test(
    test_3d_prescribed_current_injection_restart  # name
    3  # dims
    1  # nprocs
    "inputs_base_3d_prescribed_current_injection amr.restart='../test_3d_prescribed_current_injection/diags/chk000001' ${prescribed_current_waveform} ${prescribed_current_override}"  # inputs
    "analysis.py --jx-integral 1000 --jy-integral -500"  # analysis
    OFF  # checksum (quantitative restart oracle)
    test_3d_prescribed_current_injection  # dependency
)

add_warpx_test(
    test_rz_prescribed_current_injection  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_prescribed_current_injection ${prescribed_current_waveform}"  # inputs
    "analysis.py --jz -636.6197723675814 --rz-current 1000"  # analysis
    OFF  # checksum (RZ current normalization oracle)
    OFF  # dependency
)
