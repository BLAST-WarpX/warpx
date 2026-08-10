# Add pytest-based unit tests, one ctest test (= one process) per built
# dimensionality. Dimensionalities without a source subdirectory here, or that
# are not part of WarpX_DIMS, are skipped automatically.
#
foreach(_dim IN ITEMS 1 2 3 RZ RCYLINDER RSPHERE)
    add_warpx_pytest(${_dim})
endforeach()
