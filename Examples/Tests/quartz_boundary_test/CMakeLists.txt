# CMakeLists.txt for quartz boundary condition test
# This test demonstrates the new quartz boundary condition feature

# Add the test to the list of tests
add_test(
    NAME quartz_boundary_test
    COMMAND ${CMAKE_COMMAND} -E env "OMPI_COMM_WORLD_SIZE=1" $<TARGET_FILE:warpx.3d>
    --in inputs_quartz_test
    --max_step 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set test properties
set_tests_properties(quartz_boundary_test PROPERTIES
    TIMEOUT 300
    PROCESSORS 1
    LABELS "boundary_conditions;quartz;3d"
)

# Add test to the appropriate test suite
add_test_to_suite(quartz_boundary_test "boundary_conditions") 