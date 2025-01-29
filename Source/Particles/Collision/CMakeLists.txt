foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        CollisionHandler.cpp
        CollisionBase.cpp
        ScatteringProcess.cpp
    )
endforeach()

add_subdirectory(BinaryCollision)
add_subdirectory(BackgroundMCC)
add_subdirectory(BackgroundStopping)
