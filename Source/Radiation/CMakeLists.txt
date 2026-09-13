foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    if(WarpX_MATERIAL_OPACITY_HDF5)
        target_sources(lib_${SD} PRIVATE MaterialOpacityTable.cpp)
    endif()
    target_sources(lib_${SD}
      PRIVATE
        CoupledImplicitDiffusion.cpp
        CoupledMomentSource.cpp
        ImplicitDiffusion.cpp
        ImplicitMomentTransport.cpp
        MomentBalance.cpp
        MomentBoundaryExchange.cpp
        MomentTransportLedger.cpp
        OpacityTable.cpp
        ParticleImpulse.cpp
        ParticleImpulseBoundary.cpp
        RadiationTransport.cpp
        RZMomentGeometry.cpp
    )
endforeach()
