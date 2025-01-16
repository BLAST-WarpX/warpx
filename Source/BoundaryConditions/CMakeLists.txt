foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        PEC_Insulator.cpp
        PML.cpp
        WarpXEvolvePML.cpp
        WarpXFieldBoundaries.cpp
        WarpX_PEC.cpp
    )

    if(D STREQUAL "RZ")
        target_sources(lib_${SD}
          PRIVATE
            PML_RZ.cpp
        )
    endif()
endforeach()
