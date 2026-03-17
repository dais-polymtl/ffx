add_library(liballoc STATIC arena_allocator.cpp)
target_include_directories(liballoc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)