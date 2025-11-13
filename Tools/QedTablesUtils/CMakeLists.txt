# Common argument parser functions ############################################
#
# TODO: Move to ABLASTR
set(arg_parser_src
    Source/ArgParser/QedTablesArgParser.cpp
    Source/ArgParser/QedTablesArgParser.H
)


# Build QED lookup tables generator ###########################################
#
add_executable(qed_table_generator
    Source/QedTableGenerator.cpp
    ${arg_parser_src}
)
add_executable(WarpX::qed_table_generator ALIAS qed_table_generator)

target_link_libraries(qed_table_generator PRIVATE PXRMP_QED)

target_compile_features(qed_table_generator PUBLIC cxx_std_17)
set_target_properties(qed_table_generator PROPERTIES CXX_EXTENSIONS OFF)


# Build QED lookup tables reader ##############################################
#
add_executable(qed_table_reader
    Source/QedTableReader.cpp
    ${arg_parser_src}
)
add_executable(WarpX::qed_table_reader ALIAS qed_table_reader)

target_link_libraries(qed_table_reader PRIVATE PXRMP_QED)

target_compile_features(qed_table_reader PUBLIC cxx_std_17)
set_target_properties(qed_table_reader PROPERTIES CXX_EXTENSIONS OFF)
