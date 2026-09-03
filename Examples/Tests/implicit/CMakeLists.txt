# Add tests (alphabetical order) ##############################################
#

# The SINGLE/SP-PDP solver values below are precision-qualified test
# contracts, not generic solver defaults.
set(implicit_pinch_checksum "analysis_default_regression.py --path diags/diag1000020")
if(NOT WarpX_PRECISION STREQUAL "DOUBLE")
    set(implicit_pinch_checksum OFF)
endif()
if(WarpX_PRECISION STREQUAL "SINGLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_1d_theta_implicit_planar_pinch  # name
        1  # dims
        2  # nprocs
        "inputs_test_1d_theta_implicit_planar_pinch newton.relative_tolerance=5.0e-7 newton.require_convergence=1"  # inputs
        "analysis_pinch.py diags/diag1000020 --precision ${WarpX_PRECISION}"  # analysis
        "${implicit_pinch_checksum}"  # checksum
        OFF  # dependency
    )
else()
    add_warpx_test(
        test_1d_theta_implicit_planar_pinch  # name
        1  # dims
        2  # nprocs
        inputs_test_1d_theta_implicit_planar_pinch  # inputs
        "analysis_pinch.py diags/diag1000020 --precision ${WarpX_PRECISION}"  # analysis
        "${implicit_pinch_checksum}"  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_1d_semi_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_semi_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

if(AMReX_PETSC)
    if(WarpX_PRECISION STREQUAL "SINGLE"
       AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
        # SP/PDP curl, RCYLINDER, and RZ cases remain unregistered until they
        # are precision-qualified.
        add_warpx_test(
            test_2d_theta_implicit_planar_pinch  # name
            2  # dims
            2  # nprocs
            "inputs_test_2d_theta_implicit_planar_pinch newton.relative_tolerance=5.0e-7 newton.require_convergence=1"  # inputs
            "analysis_pinch.py diags/diag1000020 --precision ${WarpX_PRECISION}"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    else()
    add_warpx_test(
        test_2d_curl_curl_petsc_pc  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_curl_curl_petsc_pc  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1001000"  # analysis
        "analysis_default_regression.py --path diags/diag1001000"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_curl_curl_petsc_pc  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_theta_implicit_planar_pinch  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_theta_implicit_planar_pinch  # inputs
        "analysis_pinch.py diags/diag1000020 --precision ${WarpX_PRECISION}"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_theta_implicit_dynamic_pinch  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_theta_implicit_dynamic_pinch  # inputs
        "analysis_pinch.py diags/diag1000020 --precision ${WarpX_PRECISION}"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_theta_implicit_dynamic_pinch  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_theta_implicit_dynamic_pinch  # inputs
        "analysis_pinch.py diags/diag1000010 --precision ${WarpX_PRECISION}"  # analysis
        "analysis_default_regression.py --path diags/diag1000010"  # checksum
        OFF  # dependency
    )
    endif()
endif()

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

# CTest's PASS_REGULAR_EXPRESSION does not require the commanded process to
# fail, so use a small wrapper that checks both exit status and output.
function(add_implicit_command_contract_test name inputs expected outcome)
    if(NOT 2 IN_LIST WarpX_DIMS OR NOT WarpX_APP)
        return()
    endif()

    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()
    separate_arguments(input_args UNIX_COMMAND "${inputs}")
    set(failure_guard)
    if("${outcome}" STREQUAL "--expect-failure")
        list(APPEND failure_guard --forbid "STEP 1 ends")
    endif()
    set(test_working_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
    file(MAKE_DIRECTORY "${test_working_dir}")
    add_test(
        NAME ${name}.run
        COMMAND
            ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/check_command_output.py
            ${outcome}
            --expect "${expected}"
            ${failure_guard}
            --
            $<TARGET_FILE:app_2d>
            ${input_args}
            "amrex.abort_on_unused_inputs = 1"
            "amrex.throw_exception = 1"
            "amrex.the_arena_init_size = 0"
            "warpx.always_warn_immediately = 1"
            "warpx.do_dynamic_scheduling = 0"
            "warpx.serialize_initial_conditions = 1"
            ${ARGN}
        WORKING_DIRECTORY "${test_working_dir}"
    )
    set_tests_properties(
        ${name}.run
        PROPERTIES
            ENVIRONMENT
                "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/;OMP_NUM_THREADS=1"
    )
endfunction()

if(WarpX_PRECISION STREQUAL "SINGLE")
    set(jfnk_default_output "Newton JFNK epsilon:        0.000345")
else()
    set(jfnk_default_output "Newton JFNK epsilon:        1e-06")
endif()
add_implicit_command_contract_test(
    test_2d_theta_implicit_jfnk_default
    "inputs_test_2d_theta_implicit_jfnk_vandb max_step=0"
    "${jfnk_default_output}"
    --expect-success
)

add_implicit_command_contract_test(
    test_2d_theta_implicit_jfnk_epsilon_rejected
    "inputs_test_2d_theta_implicit_jfnk_vandb max_step=1 newton.jfnk_epsilon=0"
    "newton.jfnk_epsilon must be finite and greater than"
    --expect-failure
)

add_implicit_command_contract_test(
    test_2d_theta_implicit_linear_failure_rejected
    "inputs_test_2d_theta_implicit_jfnk_vandb max_step=1 newton.require_convergence=0 gmres.max_iterations=1"
    "Linear solver status is 1"
    --expect-failure
)

add_implicit_command_contract_test(
    test_2d_theta_implicit_restart_rejected
    "inputs_test_2d_theta_implicit_jfnk_vandb max_step=1 gmres.restart_length=0"
    "NewtonSolver: gmres.restart_length must be greater than zero"
    --expect-failure
)

if(AMReX_PETSC)
    add_implicit_command_contract_test(
        test_2d_theta_implicit_petsc_linear_failure_rejected
        "inputs_test_2d_theta_implicit_planar_pinch max_step=1 newton.require_convergence=0 gmres.max_iterations=1 newton.verbose=0 gmres.verbose_int=0 pc_petsc.verbose=0"
        "Linear solver status is -"
        --expect-failure
    )

    add_implicit_command_contract_test(
        test_2d_theta_implicit_petsc_restart_configured
        "inputs_test_2d_curl_curl_petsc_pc max_step=1 gmres.restart_length=7 newton.verbose=0 gmres.verbose_int=0 pc_petsc.verbose=0"
        "restart=7,"
        --expect-success -ksp_view
    )

    add_implicit_command_contract_test(
        test_2d_theta_implicit_petsc_restart_cli_override
        "inputs_test_2d_curl_curl_petsc_pc max_step=1 gmres.restart_length=7 newton.verbose=0 gmres.verbose_int=0 pc_petsc.verbose=0"
        "restart=5,"
        --expect-success -ksp_view -ksp_gmres_restart 5
    )

    add_implicit_command_contract_test(
        test_2d_theta_implicit_petsc_restart_rejected
        "inputs_test_2d_curl_curl_petsc_pc max_step=1 gmres.restart_length=0 newton.verbose=0 gmres.verbose_int=0 pc_petsc.verbose=0"
        "NewtonSolver: gmres.restart_length must be greater than zero"
        --expect-failure -ksp_view
    )
endif()

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_cropping  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_cropping  # inputs
    "analysis_vandb_jfnk_2d_cropping.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_filtered  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_filtered  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_picmi.py  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

# The SINGLE/SP-PDP solver values below are precision-qualified test
# contracts, not generic solver defaults.
set(implicit_symmetry_checksum "analysis_default_regression.py --path diags/diag1000400")
if(NOT WarpX_PRECISION STREQUAL "DOUBLE")
    set(implicit_symmetry_checksum OFF)
endif()
if(WarpX_PRECISION STREQUAL "SINGLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_2d_theta_implicit_symmetry  # name
        2  # dims
        2  # nprocs
        "inputs_test_2d_theta_implicit_symmetry newton.relative_tolerance=1.0e-6 newton.require_convergence=1 implicit_evolve.particle_tolerance=1.0e-10 warpx.abort_on_warning_threshold=medium"  # inputs
        "analysis_implicit.py diags/diag1000400 --precision ${WarpX_PRECISION}"  # analysis
        "${implicit_symmetry_checksum}"  # checksum
        OFF  # dependency
    )
else()
    add_warpx_test(
        test_2d_theta_implicit_symmetry  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_theta_implicit_symmetry  # inputs
        "analysis_implicit.py diags/diag1000400 --precision ${WarpX_PRECISION}"  # analysis
        "${implicit_symmetry_checksum}"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_2d_theta_implicit_strang_psatd  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_theta_implicit_strang_psatd  # inputs
        "analysis_2d_psatd.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )
endif()
