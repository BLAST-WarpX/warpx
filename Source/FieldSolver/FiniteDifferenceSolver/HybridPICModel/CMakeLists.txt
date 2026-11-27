foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
      HybridPICModel.cpp
      HybridPICCheckpoint.cpp
      ElectronHeatConduction.cpp
      NonlinearHeatConduction.cpp
      HybridElectronConduction.cpp
      ExternalVectorPotential.cpp
    )
    if(D STREQUAL "RZ")
        target_sources(lib_${SD} PRIVATE
            RZPressureInduction.cpp
            RZPressureInductionStage.cpp
            NativeRZPressureStage.cpp
            NativeRZMagneticPressure.cpp
            NativeRZMagneticPressureRuntime.cpp)
    endif()
endforeach()
