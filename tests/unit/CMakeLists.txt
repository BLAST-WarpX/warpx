# Add the pytest-based unit tests: one ctest test, that is one process, per
# built dimensionality. The tests themselves are dimensionality-agnostic and
# skip what does not apply to the geometry they run in.
#
foreach(_dim IN LISTS WarpX_DIMS)
    add_warpx_pytest(${_dim})
endforeach()
