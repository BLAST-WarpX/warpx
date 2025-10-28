# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_id_cpu_read_picmi  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_id_cpu_read_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_runtime_components_picmi  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_runtime_components_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

# FIXME
add_warpx_test(
    test_2d_runtime_components_picmi_restart  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_runtime_components_picmi.py amr.restart='../test_2d_runtime_components_picmi/diags/chk000005'"  # inputs
    OFF  #"analysis_default_restart.py diags/diag1000010"  # analysis
    OFF  #"analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_runtime_components_picmi  # dependency
)

add_warpx_test(
    test_3d_acceleration  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_acceleration  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_acceleration_restart  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_acceleration_restart  # inputs
    "analysis_default_restart.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_3d_acceleration  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_3d_acceleration_psatd  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_acceleration_psatd  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000010"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_acceleration_psatd_restart  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_acceleration_psatd_restart  # inputs
        "analysis_default_restart.py diags/diag1000010"  # analysis
        "analysis_default_regression.py --path diags/diag1000010"  # checksum
        test_3d_acceleration_psatd  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_acceleration_psatd_time_avg  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_acceleration_psatd_time_avg  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000010"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_FFT)
    add_warpx_test(
        test_3d_acceleration_psatd_time_avg_restart  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_acceleration_psatd_time_avg_restart  # inputs
        "analysis_default_restart.py diags/diag1000010"  # analysis
        "analysis_default_regression.py --path diags/diag1000010"  # checksum
        test_3d_acceleration_psatd_time_avg  # dependency
    )
endif()
