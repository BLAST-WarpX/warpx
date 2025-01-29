foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(ablastr_${SD}
      PRIVATE
        Communication.cpp
        SignalHandling.cpp
        TextMsg.cpp
        UsedInputsFile.cpp
    )
endforeach()

add_subdirectory(msg_logger)
add_subdirectory(text)
add_subdirectory(timer)
