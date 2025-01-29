foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        ComputeDivE.cpp
        EvolveB.cpp
        EvolveBPML.cpp
        EvolveE.cpp
        EvolveEPML.cpp
        EvolveF.cpp
        EvolveFPML.cpp
        EvolveG.cpp
        EvolveECTRho.cpp
        FiniteDifferenceSolver.cpp
        HybridPICSolveE.cpp
        MacroscopicEvolveE.cpp
        ApplySilverMuellerBoundary.cpp
    )
endforeach()

add_subdirectory(HybridPICModel)
add_subdirectory(MacroscopicProperties)
