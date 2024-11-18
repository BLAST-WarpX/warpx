# delete all test files except backtrace
file(GLOB test_files ${CMAKE_ARGV3}/*)
foreach(file ${test_files})
    if(NOT ${file} MATCHES "Backtrace*")
        execute_process(COMMAND ${CMAKE_COMMAND} -E rm -r ${file})
    endif()
endforeach()
