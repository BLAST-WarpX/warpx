foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        ExternalField.cpp
        GetTemperature.cpp
        GetVelocity.cpp
        InjectorDensity.cpp
        InjectorMomentum.cpp
        PlasmaInjector.cpp
        TemperatureProperties.cpp
        VelocityProperties.cpp
        WarpXAMReXInit.cpp
        WarpXInit.cpp
        WarpXInitData.cpp
    )
endforeach()

add_subdirectory(DivCleaner)
