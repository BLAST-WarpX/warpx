add_warpx_test(
    test_3d_iterated_static_levels  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_iterated_static_levels  # inputs
    "analysis_iterated_static_levels.py diags/diag1000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)
