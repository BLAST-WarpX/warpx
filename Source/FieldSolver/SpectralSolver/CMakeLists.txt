foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        SpectralFieldData.cpp
        SpectralKSpace.cpp
        SpectralSolver.cpp
    )

    if(D STREQUAL "RZ")
        target_sources(lib_${SD}
          PRIVATE
            SpectralSolverRZ.cpp
            SpectralFieldDataRZ.cpp
            SpectralKSpaceRZ.cpp
            SpectralBinomialFilter.cpp
        )
    endif()
endforeach()

if(RZ IN_LIST WarpX_DIMS)
    add_subdirectory(SpectralHankelTransform)
endif()
add_subdirectory(SpectralAlgorithms)
