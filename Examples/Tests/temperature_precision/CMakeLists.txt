add_warpx_test(
    test_1d_temperature_precision
    1
    1
    inputs_test_1d_temperature_precision
    "analysis.py --precision ${WarpX_PRECISION}"
    OFF
    OFF
)
