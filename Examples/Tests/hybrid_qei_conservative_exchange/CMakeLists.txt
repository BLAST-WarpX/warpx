# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_hybrid_qei_conservative_exchange  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_hybrid_qei_conservative_exchange  # inputs
    "analysis.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic exchange and conservative ledgers are checked)
    OFF  # dependency
)
