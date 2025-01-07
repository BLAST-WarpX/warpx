foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})

    target_sources(ablastr_${SD}
      PRIVATE
        MultiFabRegister.cpp
    )

    if(ABLASTR_FFT AND D EQUAL 3)
        target_sources(ablastr_${SD}
          PRIVATE
            IntegratedGreenFunctionSolver.cpp
        )
    endif()
endforeach()
