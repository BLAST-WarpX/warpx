foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        Interpolate.cpp
        ParticleUtils.cpp
        SpeciesUtils.cpp
        WarpXMovingWindow.cpp
        WarpXUtil.cpp
        WarpXVersion.cpp
    )
endforeach()

add_subdirectory(Logo)
add_subdirectory(Parser)
