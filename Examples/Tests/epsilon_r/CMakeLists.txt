# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_epsilon_r  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_epsilon_r  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

foreach(method IN ITEMS backwardeuler laxwendroff)
    foreach(parameter IN ITEMS epsilon_r c_light_factor)
        foreach(factor IN ITEMS 1 2 10)
            add_warpx_test(
                test_2d_epsilon_r_conductivity_${method}_${parameter}_${factor}
                2
                1
                "inputs_base_conductivity.py --method ${method} --parameter ${parameter} --factor ${factor}"
                OFF  # analytic relaxation and driven Ohmic-response assertions in the input
                OFF  # checksum
                OFF  # dependency
            )
        endforeach()
    endforeach()
    add_warpx_test(
        test_2d_epsilon_r_conductivity_${method}_default
        2
        1
        "inputs_base_conductivity.py --method ${method} --parameter default"
        OFF
        OFF
        OFF
    )
endforeach()

foreach(condition IN ITEMS both small_factor small_epsilon overflow)
    add_warpx_test(
        test_2d_epsilon_r_guard_${condition}
        2
        1
        "inputs_base_parameter_guard.py --case ${condition}"
        OFF
        OFF
        OFF
    )
endforeach()

foreach(parameter IN ITEMS epsilon_r c_light_factor)
    foreach(factor IN ITEMS 1 2 10)
        add_warpx_test(
            test_2d_epsilon_r_wave_${parameter}_${factor}
            2
            1
            "inputs_base_wave_speed.py --parameter ${parameter} --factor ${factor}"
            OFF
            OFF
            OFF
        )
    endforeach()
endforeach()
add_warpx_test(
    test_2d_epsilon_r_wave_default
    2
    1
    "inputs_base_wave_speed.py --parameter default"
    OFF
    OFF
    OFF
)
