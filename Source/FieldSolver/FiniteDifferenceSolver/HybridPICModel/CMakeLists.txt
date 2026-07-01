foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
      ConformalEctLSQ.cpp
      EBJBoundary.cpp
      HybridPICModel.cpp
      ExternalVectorPotential.cpp
      MarderCorrection.cpp
    )
endforeach()
