add_library(libbench STATIC benchmark.cpp)
target_include_directories(libbench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(libbench PUBLIC libquery PUBLIC libplan PUBLIC libserder)
