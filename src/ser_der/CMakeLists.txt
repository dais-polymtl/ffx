add_library(libserder STATIC serializer.cpp table_metadata.cpp)
target_include_directories(libserder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(libserder PRIVATE Threads::Threads PRIVATE libtable)
