foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        CellCenterFunctor.cpp
        DivBFunctor.cpp
        DivEFunctor.cpp
        EBCoveredFunctor.cpp
        JFunctor.cpp
        JdispFunctor.cpp
        RhoFunctor.cpp
        PartPerCellFunctor.cpp
        PartPerGridFunctor.cpp
        ProcessNumberFunctor.cpp
        BackTransformFunctor.cpp
        BackTransformParticleFunctor.cpp
        ParticleReductionFunctor.cpp
        PhiFunctor.cpp
        TemperatureFunctor.cpp
    )
endforeach()
