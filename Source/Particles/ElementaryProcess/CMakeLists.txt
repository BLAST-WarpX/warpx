foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        Ionization.cpp
    )

    if(WarpX_QED)
        target_sources(lib_${SD}
          PRIVATE
            QEDPairGeneration.cpp
            QEDPhotonEmission.cpp
        )
    endif()
endforeach()

if(WarpX_QED)
    add_subdirectory(QEDInternals)
endif()
