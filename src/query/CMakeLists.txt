add_library(libquery STATIC 
    query.cpp
    query_rule_parser.cpp
    predicate_parser.cpp
)
target_include_directories(libquery PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(libquery PUBLIC re2::re2)
