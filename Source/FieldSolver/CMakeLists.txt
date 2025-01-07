foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        WarpXPushFieldsEM.cpp
        WarpXPushFieldsHybridPIC.cpp
        WarpX_QED_Field_Pushers.cpp
        WarpXSolveFieldsES.cpp
    )
endforeach()

add_subdirectory(ElectrostaticSolvers)
add_subdirectory(FiniteDifferenceSolver)
add_subdirectory(MagnetostaticSolver)
add_subdirectory(ImplicitSolvers)
if(WarpX_FFT)
    add_subdirectory(SpectralSolver)
endif()
