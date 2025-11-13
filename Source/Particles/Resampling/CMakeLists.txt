foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        Resampling.cpp
        ResamplingTrigger.cpp
        LevelingThinning.cpp
        VelocityCoincidenceThinning.cpp
    )
endforeach()
