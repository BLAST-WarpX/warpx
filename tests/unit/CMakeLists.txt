# Add pytest-based unit tests, one ctest test (= one process) per built
# dimensionality. Dimensionalities without a source subdirectory here are
# skipped automatically.
#
foreach(_dim IN LISTS WarpX_DIMS)
    add_warpx_pytest(${_dim})
endforeach()
