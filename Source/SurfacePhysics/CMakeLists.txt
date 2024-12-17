foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    if (WarpX_SURFACE_PHYSICS)
        target_sources(lib_${SD}
          PRIVATE
            SurfacePhysicsBase.cpp
        )
    endif()
endforeach()
