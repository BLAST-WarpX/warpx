foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        TimeTracker.cpp
        GuardCellManager.cpp
        WarpXComm.cpp
        WarpXRegrid.cpp
        WarpXSumGuardCells.cpp
    )
endforeach()
