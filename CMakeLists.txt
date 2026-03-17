cmake_minimum_required(VERSION 3.5)

project(ffx VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Add project subdirectories
add_subdirectory(src/table)
add_subdirectory(src/ser_der)
add_subdirectory(src/query)
add_subdirectory(src/flock)
add_subdirectory(src/operator)
add_subdirectory(src/plan)
add_subdirectory(src/benchmark)

# Find required packages
find_package(Threads REQUIRED)

# Add GoogleTest
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0
)
# Add Abseil (required by RE2)
FetchContent_Declare(
        absl
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20250127.0
)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE) # Try enabling install to match re2's expectation if it exports

# Add Google RE2
FetchContent_Declare(
        re2
        GIT_REPOSITORY https://github.com/google/re2.git
        GIT_TAG 2023-09-01
)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(googletest absl re2)
include(GoogleTest)

# Define build options
option(STORAGE_TO_VECTOR_MEMCPY_PTR_ALIAS "Disable pointer aliasing" OFF)
option(BIT_ARRAY_AS_FILTER "Use bit arrays for selection" OFF)
option(ENABLE_SELECTION_POS_VECTOR "Use selection position for selection" OFF)

# Check for incompatible options
if (BIT_ARRAY_AS_FILTER AND ENABLE_SELECTION_POS_VECTOR)
    message(FATAL_ERROR "Cannot enable both BIT_ARRAY_AS_FILTER and ENABLE_SELECTION_POS_VECTOR at the same time")
endif()

# Main executables
add_executable(table_serializer src/table_serializer.cpp)
add_executable(query_eval_exec src/query_eval_exec.cpp)

# Link libraries to executables
target_link_libraries(table_serializer PUBLIC libserder)
target_link_libraries(query_eval_exec PUBLIC libbench)

# Helper function to set compiler options and definitions
function(configure_target target)
    # Set compiler options based on build type
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Abseil (via RE2) uses Clang nullability attributes; flock headers may have unused parameters in interfaces.
        target_compile_options(${target} PRIVATE -g3 -Wall -Wextra -Wpedantic -Werror -Wno-nullability-extension
                -Wno-unused-parameter)
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${target} PRIVATE -Ofast -march=native)
    endif()

    # Set compile definitions based on options
    if (STORAGE_TO_VECTOR_MEMCPY_PTR_ALIAS)
        target_compile_definitions(${target} PRIVATE STORAGE_TO_VECTOR_MEMCPY_PTR_ALIAS)
    endif()

    if (BIT_ARRAY_AS_FILTER)
        target_compile_definitions(${target} PRIVATE BIT_ARRAY_AS_FILTER)
    endif()

    if (ENABLE_SELECTION_POS_VECTOR)
        target_compile_definitions(${target} PRIVATE ENABLE_SELECTION_POS_VECTOR)
    endif()
endfunction()

# Configure main executables
configure_target(table_serializer)
configure_target(query_eval_exec)

# Add test executables based on build options
if (NOT ENABLE_SELECTION_POS_VECTOR)
    # Add bitmask tests
    add_executable(bitmask_tests tests/bitmask_tests.cpp)
    target_link_libraries(bitmask_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(bitmask_tests)
    gtest_discover_tests(bitmask_tests)
    message(STATUS "Added bitmask_tests (using default bitmask selection)")

    # Add iterator tests
    add_executable(iterator_tests tests/iterator_tests.cpp)
    target_link_libraries(iterator_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(iterator_tests)
    gtest_discover_tests(iterator_tests)
    message(STATUS "Added iterator_tests (using default bitmask selection)")

    # Add batched f-tree iterator tests
    add_executable(ftree_batch_iterator_tests tests/ftree_batch_iterator_tests.cpp)
    target_link_libraries(ftree_batch_iterator_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(ftree_batch_iterator_tests)
    gtest_discover_tests(ftree_batch_iterator_tests)
    message(STATUS "Added ftree_batch_iterator_tests (using default bitmask selection)")

    # Add streaming map tests
    add_executable(streaming_map_tests tests/streaming_map_tests.cpp)
    target_link_libraries(streaming_map_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(streaming_map_tests)
    gtest_discover_tests(streaming_map_tests)
    message(STATUS "Added streaming_map_tests (using default bitmask selection)")

    # Add FTree reconstructor tests (previously cartesian_product_tests)
    add_executable(cartesian_product_tests tests/cartesian_product_tests.cpp)
    target_link_libraries(cartesian_product_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(cartesian_product_tests)
    gtest_discover_tests(cartesian_product_tests)
    message(STATUS "Added cartesian_product_tests (using default bitmask selection)")

    # Add sink packed tests
    add_executable(sink_packed_tests tests/sink_packed_tests.cpp)
    target_link_libraries(sink_packed_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(sink_packed_tests)
    gtest_discover_tests(sink_packed_tests)
    message(STATUS "Added sink_packed_tests (using default bitmask selection)")

    # Add sink min tests (string support)
    add_executable(sink_min_tests tests/sink_min_tests.cpp)
    target_link_libraries(sink_min_tests PUBLIC liboperator libtable GTest::gtest_main re2::re2)
    configure_target(sink_min_tests)
    gtest_discover_tests(sink_min_tests)
    message(STATUS "Added sink_min_tests (using default bitmask selection)")

    add_executable(sink_min_partial_tests tests/sink_min_partial_tests.cpp)
    target_link_libraries(sink_min_partial_tests PUBLIC liboperator libtable GTest::gtest_main re2::re2)
    configure_target(sink_min_partial_tests)
    gtest_discover_tests(sink_min_partial_tests)
    message(STATUS "Added sink_min_partial_tests")

    add_executable(flat_join_intersection_tests tests/flat_join_intersection_tests.cpp)
    target_link_libraries(flat_join_intersection_tests PUBLIC libplan liboperator libtable libquery GTest::gtest_main)
    configure_target(flat_join_intersection_tests)
    gtest_discover_tests(flat_join_intersection_tests)
    message(STATUS "Added flat_join_intersection_tests (using default bitmask selection)")

    add_executable(cycle_handling_tests tests/cycle_handling_tests.cpp)
    target_link_libraries(cycle_handling_tests PUBLIC libplan liboperator libtable libquery GTest::gtest_main)
    configure_target(cycle_handling_tests)
    gtest_discover_tests(cycle_handling_tests)
    message(STATUS "Added cycle_handling_tests (using default bitmask selection)")

    # Add string support tests
    add_executable(ffx_str_t_tests tests/ffx_str_t_tests.cpp)
    target_link_libraries(ffx_str_t_tests PUBLIC libtable GTest::gtest_main)
    configure_target(ffx_str_t_tests)
    gtest_discover_tests(ffx_str_t_tests)
    message(STATUS "Added ffx_str_t_tests")

    add_executable(string_pool_tests tests/string_pool_tests.cpp)
    target_link_libraries(string_pool_tests PUBLIC libtable GTest::gtest_main)
    configure_target(string_pool_tests)
    gtest_discover_tests(string_pool_tests)
    message(STATUS "Added string_pool_tests")

    add_executable(string_dictionary_ser_tests tests/string_dictionary_ser_tests.cpp)
    target_link_libraries(string_dictionary_ser_tests PUBLIC libtable GTest::gtest_main)
    configure_target(string_dictionary_ser_tests)
    gtest_discover_tests(string_dictionary_ser_tests)
    message(STATUS "Added string_dictionary_ser_tests")

    add_executable(string_dictionary_integration_tests tests/string_dictionary_integration_tests.cpp)
    target_link_libraries(string_dictionary_integration_tests PUBLIC libserder libtable GTest::gtest_main)
    configure_target(string_dictionary_integration_tests)
    gtest_discover_tests(string_dictionary_integration_tests)
    message(STATUS "Added string_dictionary_integration_tests")

    add_executable(string_hash_table_tests tests/string_hash_table_tests.cpp)
    target_link_libraries(string_hash_table_tests PUBLIC libtable GTest::gtest_main)
    configure_target(string_hash_table_tests)
    gtest_discover_tests(string_hash_table_tests)
    message(STATUS "Added string_hash_table_tests")

    add_executable(string_dictionary_tests tests/string_dictionary_tests.cpp)
    target_link_libraries(string_dictionary_tests PUBLIC libtable GTest::gtest_main)
    configure_target(string_dictionary_tests)
    gtest_discover_tests(string_dictionary_tests)
    message(STATUS "Added string_dictionary_tests")

    add_executable(string_dictionary_mt_tests tests/string_dictionary_mt_tests.cpp)
    target_link_libraries(string_dictionary_mt_tests PUBLIC libtable GTest::gtest_main Threads::Threads)
    configure_target(string_dictionary_mt_tests)
    gtest_discover_tests(string_dictionary_mt_tests)
    message(STATUS "Added string_dictionary_mt_tests")

    add_executable(dictionary_pipeline_e2e_tests tests/dictionary_pipeline_e2e_tests.cpp)
    target_link_libraries(dictionary_pipeline_e2e_tests PUBLIC libplan libquery libserder libtable liboperator GTest::gtest_main)
    configure_target(dictionary_pipeline_e2e_tests)
    gtest_discover_tests(dictionary_pipeline_e2e_tests)
    message(STATUS "Added dictionary_pipeline_e2e_tests")

    add_executable(mid_pipeline_llm_e2e_tests tests/mid_pipeline_llm_e2e_tests.cpp)
    target_link_libraries(mid_pipeline_llm_e2e_tests PUBLIC libplan libquery libserder libtable liboperator GTest::gtest_main)
    configure_target(mid_pipeline_llm_e2e_tests)
    gtest_discover_tests(mid_pipeline_llm_e2e_tests)
    message(STATUS "Added mid_pipeline_llm_e2e_tests")

    add_executable(ser_des_str_tests tests/ser_des_str_tests.cpp)
    target_link_libraries(ser_des_str_tests PUBLIC libserder libtable GTest::gtest_main)
    configure_target(ser_des_str_tests)
    gtest_discover_tests(ser_des_str_tests)

    add_executable(table_serializer_tests tests/table_serializer_tests.cpp)
    target_link_libraries(table_serializer_tests PUBLIC libserder libtable GTest::gtest_main)
    configure_target(table_serializer_tests)
    gtest_discover_tests(table_serializer_tests)
    message(STATUS "Added table_serializer_tests")
    message(STATUS "Added ser_des_str_tests")

    add_executable(ftree_ancestor_finder_tests tests/ftree_ancestor_finder_tests.cpp)
    target_link_libraries(ftree_ancestor_finder_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(ftree_ancestor_finder_tests)
    gtest_discover_tests(ftree_ancestor_finder_tests)
    message(STATUS "Added ftree_ancestor_finder_tests (using default bitmask selection)")

    # Add predicate parser tests
    add_executable(predicate_parser_tests tests/predicate_parser_tests.cpp)
    target_link_libraries(predicate_parser_tests PUBLIC libquery GTest::gtest_main re2::re2)
    configure_target(predicate_parser_tests)
    gtest_discover_tests(predicate_parser_tests)
    message(STATUS "Added predicate_parser_tests")

    add_executable(predicate_eval_string_tests tests/predicate_eval_string_tests.cpp)
    target_link_libraries(predicate_eval_string_tests PUBLIC libquery liboperator libtable GTest::gtest_main re2::re2)
    configure_target(predicate_eval_string_tests)
    gtest_discover_tests(predicate_eval_string_tests)
    message(STATUS "Added predicate_eval_string_tests")
    add_executable(re2_like_tests tests/re2_like_tests.cpp)
    target_link_libraries(re2_like_tests PUBLIC GTest::gtest_main re2::re2)
    configure_target(re2_like_tests)
    gtest_discover_tests(re2_like_tests)
    message(STATUS "Added re2_like_tests")

    # Add predicated operators tests
    add_executable(predicated_operators_tests tests/predicated_operators_tests.cpp)
    target_link_libraries(predicated_operators_tests PUBLIC liboperator libtable libquery libplan GTest::gtest_main)
    configure_target(predicated_operators_tests)
    gtest_discover_tests(predicated_operators_tests)
    message(STATUS "Added predicated_operators_tests")

    # Cardinality and state sharing tests
    add_executable(cardinality_tests tests/cardinality_tests.cpp)
    target_link_libraries(cardinality_tests PUBLIC libtable liboperator GTest::gtest_main)
    configure_target(cardinality_tests)
    gtest_discover_tests(cardinality_tests)
    message(STATUS "Added cardinality_tests")

    # Add adj list manager tests
    add_executable(adj_list_manager_tests tests/adj_list_manager_tests.cpp)
    target_link_libraries(adj_list_manager_tests PUBLIC liboperator libtable libserder GTest::gtest_main)
    configure_target(adj_list_manager_tests)
    gtest_discover_tests(adj_list_manager_tests)
    message(STATUS "Added adj_list_manager_tests")

    # Add table metadata tests
    add_executable(table_metadata_tests tests/table_metadata_tests.cpp)
    target_link_libraries(table_metadata_tests PUBLIC libserder GTest::gtest_main)
    configure_target(table_metadata_tests)
    gtest_discover_tests(table_metadata_tests)
    message(STATUS "Added table_metadata_tests")

    # Add column serializer tests
    add_executable(column_serializer_tests tests/column_serializer_tests.cpp)
    target_link_libraries(column_serializer_tests PUBLIC libserder libtable GTest::gtest_main)
    configure_target(column_serializer_tests)
    gtest_discover_tests(column_serializer_tests)
    message(STATUS "Added column_serializer_tests")

    # Add AI serializer tests
    add_executable(ai_serializer_tests tests/ai_operator/ai_serializer_tests.cpp)
    target_link_libraries(ai_serializer_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(ai_serializer_tests)
    gtest_discover_tests(ai_serializer_tests)
    message(STATUS "Added ai_serializer_tests")

    # Add adjacency list builder tests
    add_executable(adj_list_builder_tests tests/adj_list_builder_tests.cpp)
    target_link_libraries(adj_list_builder_tests PUBLIC libtable GTest::gtest_main)
    configure_target(adj_list_builder_tests)
    gtest_discover_tests(adj_list_builder_tests)
    message(STATUS "Added adj_list_builder_tests")

    # Add loaded table tests
    add_executable(loaded_table_tests tests/loaded_table_tests.cpp)
    target_link_libraries(loaded_table_tests PUBLIC libtable libserder GTest::gtest_main)
    configure_target(loaded_table_tests)
    gtest_discover_tests(loaded_table_tests)
    message(STATUS "Added loaded_table_tests")

    # Add serialization/deserialization integration tests
    add_executable(ser_des_integration_tests tests/ser_des_integration_tests.cpp)
    target_link_libraries(ser_des_integration_tests PUBLIC libserder libtable GTest::gtest_main)
    configure_target(ser_des_integration_tests)
    gtest_discover_tests(ser_des_integration_tests)
    message(STATUS "Added ser_des_integration_tests")

    # Add table loader integration tests
    add_executable(table_loader_integration_tests tests/table_loader_integration_tests.cpp)
    target_link_libraries(table_loader_integration_tests PUBLIC libtable liboperator libquery libserder GTest::gtest_main)
    configure_target(table_loader_integration_tests)
    gtest_discover_tests(table_loader_integration_tests)
    message(STATUS "Added table_loader_integration_tests")

    # Add extended AdjListManager tests
    add_executable(adj_list_manager_extended_tests tests/adj_list_manager_extended_tests.cpp)
    target_link_libraries(adj_list_manager_extended_tests PUBLIC liboperator libtable GTest::gtest_main)
    configure_target(adj_list_manager_extended_tests)
    gtest_discover_tests(adj_list_manager_extended_tests)
    message(STATUS "Added adj_list_manager_extended_tests")

    # Add edge case tests
    add_executable(edge_case_tests tests/edge_case_tests.cpp)
    target_link_libraries(edge_case_tests PUBLIC libtable liboperator libquery libserder GTest::gtest_main)
    configure_target(edge_case_tests)
    gtest_discover_tests(edge_case_tests)
    message(STATUS "Added edge_case_tests")

    # Add Datalog mapping tests
    add_executable(datalog_mapping_tests tests/datalog_mapping_tests.cpp)
    target_link_libraries(datalog_mapping_tests PUBLIC libtable libquery libserder liboperator GTest::gtest_main)
    configure_target(datalog_mapping_tests)
    gtest_discover_tests(datalog_mapping_tests)
    message(STATUS "Added datalog_mapping_tests")

    # Add Datalog table loading tests
    add_executable(datalog_table_loading_tests tests/datalog_table_loading_tests.cpp)
    target_link_libraries(datalog_table_loading_tests PUBLIC libtable liboperator libquery libserder GTest::gtest_main)
    configure_target(datalog_table_loading_tests)
    gtest_discover_tests(datalog_table_loading_tests)
    message(STATUS "Added datalog_table_loading_tests")

    # Add Datalog end-to-end tests
    add_executable(datalog_e2e_tests tests/datalog_e2e_tests.cpp)
    target_link_libraries(datalog_e2e_tests PUBLIC libtable liboperator libquery libserder GTest::gtest_main)
    configure_target(datalog_e2e_tests)
    gtest_discover_tests(datalog_e2e_tests)
    message(STATUS "Added datalog_e2e_tests")

    # Add operator schema lookup tests
    add_executable(operator_schema_lookup_tests tests/operator_schema_lookup_tests.cpp)
    target_link_libraries(operator_schema_lookup_tests PUBLIC liboperator libtable libquery GTest::gtest_main)
    configure_target(operator_schema_lookup_tests)
    gtest_discover_tests(operator_schema_lookup_tests)
    message(STATUS "Added operator_schema_lookup_tests")

    add_executable(query_parser_tests tests/query_parser_tests.cpp)
    target_link_libraries(query_parser_tests PUBLIC libquery GTest::gtest_main)
    configure_target(query_parser_tests)
    gtest_discover_tests(query_parser_tests)
    message(STATUS "Added query_parser_tests")

    # Add Big Integration Test
    add_executable(integration_big_test tests/integration_big/integration_big_test.cpp)
    target_link_libraries(integration_big_test PUBLIC libplan libquery libserder libtable liboperator GTest::gtest_main)
    configure_target(integration_big_test)
    
    # Automated Big Integration Test with data generation, serialization, and cleanup
    file(CHMOD "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_big/run_integration_big.sh" 
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
         
    add_test(NAME BigIntegrationTest
             COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_big/run_integration_big.sh" 
                     "${CMAKE_CURRENT_BINARY_DIR}" 
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_big")
    
    # LLM position variant tests
    foreach(POS pos2 pos3 pos4 pos5 end)
        set(TEST_NAME integration_big_llm_${POS}_test)
        add_executable(${TEST_NAME} tests/integration_big/integration_big_llm_${POS}_test.cpp)
        target_link_libraries(${TEST_NAME} PUBLIC libplan libquery libserder libtable liboperator GTest::gtest_main)
        configure_target(${TEST_NAME})
    endforeach()

    # Star topology test (ATitle→BTitle, ATitle→CTitle with LLM)
    add_executable(integration_big_star_topology_test tests/integration_big/integration_big_star_topology_test.cpp)
    target_link_libraries(integration_big_star_topology_test PUBLIC libplan libquery libserder libtable liboperator GTest::gtest_main)
    configure_target(integration_big_star_topology_test)

    message(STATUS "Added integration_big_test and BigIntegrationTest automated suite")

endif()
# Configure libraries
#configure_target(libtable)
#configure_target(libserder)
#configure_target(libquery)
#configure_target(liboperator)
#configure_target(libplan)
#configure_target(libbench)

## Add test executables based on build options
#if (NOT ENABLE_SELECTION_POS_VECTOR)
#    # Add bitmask tests
#    add_executable(bitmask_tests tests/bitmask_tests.cpp)
#    target_link_libraries(bitmask_tests PUBLIC liboperator libtable GTest::gtest_main)
#    configure_target(bitmask_tests)
#    gtest_discover_tests(bitmask_tests)
#    message(STATUS "Added bitmask_tests (using default bitmask selection)")
#endif()
#
#if (ENABLE_SELECTION_POS_VECTOR)
#    # Add selection vector tests
#    add_executable(selection_vector_tests tests/selection_vector_tests.cpp)
#    target_link_libraries(selection_vector_tests PUBLIC liboperator libtable GTest::gtest_main)
#    configure_target(selection_vector_tests)
#    gtest_discover_tests(selection_vector_tests)
#    message(STATUS "Added selection_vector_tests (using selection position vector)")
#endif()

# Print selection method being used
if (BIT_ARRAY_AS_FILTER)
    message(STATUS "Using bit array selection vector implementation")
elseif (ENABLE_SELECTION_POS_VECTOR)
    message(STATUS "Using selection position vector implementation")
else()
    message(STATUS "Using default bitmask selection vector implementation")
endif()