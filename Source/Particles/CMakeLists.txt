foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        AddPlasmaUtilities.cpp
        MultiParticleContainer.cpp
        ParticleBoundaries.cpp
        PhotonParticleContainer.cpp
        PhysicalParticleContainer.cpp
        RigidInjectedParticleContainer.cpp
        WarpXParticleContainer.cpp
        LaserParticleContainer.cpp
        ParticleBoundaryBuffer.cpp
        SpeciesPhysicalProperties.cpp
    )
endforeach()

#add_subdirectory(Algorithms)
add_subdirectory(Collision)
#add_subdirectory(Deposition)
add_subdirectory(ElementaryProcess)
add_subdirectory(Gather)
add_subdirectory(ParticleCreation)
#add_subdirectory(Pusher)
add_subdirectory(Resampling)
add_subdirectory(Sorting)
