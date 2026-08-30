foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    if(WarpX_MATERIAL_OPACITY_HDF5)
        target_sources(lib_${SD} PRIVATE MaterialOpacityTable.cpp)
    endif()
    target_sources(lib_${SD}
      PRIVATE
        OpacityTable.cpp
        RadiationTransport.cpp
    )
endforeach()
