foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(ablastr_${SD}
      PRIVATE
        FiniteDifference.cpp
    )
endforeach()

add_subdirectory(fft)
