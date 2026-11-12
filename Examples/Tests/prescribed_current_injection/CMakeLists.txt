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
    test_2d_prescribed_current_kinetic_picmi
    2
    1
    inputs_base_2d_antenna_plasma.py
    OFF
    OFF
    OFF
)

add_warpx_test(
    test_2d_prescribed_current_straddling_uniform_picmi
    2
    1
    "inputs_base_2d_antenna_plasma.py --across-interface"
    OFF
    OFF
    OFF
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
            PASS_REGULAR_EXPRESSION "impressed-current antenna, not an ion species"
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

# Exercise physical drive sizes smaller than a cell in one direction.
add_warpx_test(
    test_2d_prescribed_current_subcell
    2
    1
    "inputs_test_2d_prescribed_current_subcell ${prescribed_current_waveform} ${prescribed_current_override}"
    "analysis.py --jx-integral 750 --jy-integral 1500 --jz-integral -375"
    OFF
    OFF
)

add_warpx_test(
    test_3d_prescribed_current_subcell
    3
    1
    "inputs_test_3d_prescribed_current_subcell ${prescribed_current_waveform} ${prescribed_current_override}"
    "analysis.py --jx-integral 1250 --jy-integral -625 --check-continuity"
    OFF
    OFF
)

add_warpx_test(
    test_rz_prescribed_current_refined_shell
    RZ
    1
    "inputs_test_rz_prescribed_current_refined_shell ${prescribed_current_waveform}"
    "analysis_rz_refined_shell.py diags/diag1000001"
    OFF
    OFF
)

add_warpx_test(
    test_rz_prescribed_current_joined_uniform_picmi
    RZ
    1
    "inputs_base_rz_joined_antenna.py --uniform"
    OFF
    OFF
    OFF
)

add_warpx_test(
    test_rz_prescribed_current_joined_vacuum_picmi
    RZ
    1
    "inputs_base_rz_joined_antenna.py"
    OFF
    OFF
    OFF
)

add_warpx_test(
    test_rz_prescribed_current_joined_plasma_picmi
    RZ
    1
    "inputs_base_rz_joined_antenna.py --plasma"
    OFF
    OFF
    OFF
)

if(RZ IN_LIST WarpX_DIMS AND WarpX_PYTHON)
    # A stored kinetic population can remain inside the bore while the prescribed
    # source is regenerated on the parent mesh from its input.
    foreach(particle IN ITEMS bore parent)
        set(seed test_rz_prescribed_current_guarded_${particle}_picmi)
        set(flags "--plasma --parent-coil")
        if(particle STREQUAL "parent")
            string(APPEND flags " --parent-particle")
        endif()
        add_warpx_test(
            test_rz_prescribed_current_guarded_${particle}_picmi
            RZ
            1
            "inputs_base_rz_joined_antenna.py ${flags} --guarded-seed"
            OFF
            OFF
            OFF
        )
        add_warpx_test(
            test_rz_prescribed_current_guarded_${particle}_picmi_restart
            RZ
            1
            "inputs_base_rz_joined_antenna.py ${flags} --restart ../${seed}/diags/chk000000"
            OFF
            OFF
            ${seed}
        )
        if(particle STREQUAL "parent")
            set_tests_properties(${seed}_restart.run PROPERTIES
                PASS_REGULAR_EXPRESSION "A guarded radial checkpoint can enable fixed-shape coupling only")
        endif()
    endforeach()
    set(seed test_rz_prescribed_current_guarded_bore_picmi)
    add_warpx_test(
        test_rz_prescribed_current_guarded_active_coil_picmi_restart
        RZ
        1
        "inputs_base_rz_joined_antenna.py --plasma --parent-coil --restart ../${seed}/diags/chk000001"
        OFF
        OFF
        ${seed}
    )
    set_tests_properties(test_rz_prescribed_current_guarded_active_coil_picmi_restart.run PROPERTIES
        PASS_REGULAR_EXPRESSION "A guarded radial checkpoint can enable fixed-shape coupling only")
endif()
