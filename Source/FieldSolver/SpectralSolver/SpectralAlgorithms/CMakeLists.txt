foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        PsatdJRhomAlgorithmFirstOrder.cpp
        PsatdAlgorithmGalilean.cpp
        PsatdJRhomAlgorithmSecondOrder.cpp
        PsatdAlgorithmPml.cpp
        SpectralBaseAlgorithm.cpp
        PsatdAlgorithmComoving.cpp
    )

    if(D STREQUAL "RZ")
        target_sources(lib_${SD}
          PRIVATE
            SpectralBaseAlgorithmRZ.cpp
            PsatdAlgorithmRZ.cpp
            PsatdAlgorithmGalileanRZ.cpp
            PsatdAlgorithmPmlRZ.cpp
        )
    endif()
endforeach()
