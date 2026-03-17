add_library(libplan STATIC plan.cpp plan_tree.cpp)

target_include_directories(libplan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(libplan PUBLIC liboperator PUBLIC libtable PUBLIC libquery)
