foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        CompositeCurrentControlledPort.cpp
        CompositeYeeSurface.cpp
        FixedShapeCurrent.cpp
        CurrentControlledPort.cpp
        WarpXPushFieldsEM.cpp
        WarpXPushFieldsHybridPIC.cpp
        WarpX_QED_Field_Pushers.cpp
        WarpXSolveFieldsES.cpp
    )
    if(D STREQUAL "RZ")
        target_sources(lib_${SD} PRIVATE ConformingRadialInterface.cpp)
    endif()
endforeach()

add_subdirectory(ElectrostaticSolvers)
add_subdirectory(FiniteDifferenceSolver)
add_subdirectory(MagnetostaticSolver)
add_subdirectory(ImplicitSolvers)
if(WarpX_FFT)
    add_subdirectory(SpectralSolver)
endif()
