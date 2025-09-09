foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        SpectralBaseAlgorithm.cpp
        PsatdAlgorithmJRhomFirstOrder.cpp
        PsatdAlgorithmJRhomSecondOrder.cpp
        PsatdAlgorithmGalilean.cpp
        PsatdAlgorithmComoving.cpp
        PsatdAlgorithmPml.cpp
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
