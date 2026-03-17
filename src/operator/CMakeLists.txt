add_library(liboperator STATIC
        scan/scan.cpp
        scan/scan_synchronized.cpp
        scan/scan_predicated.cpp
        scan/scan_synchronized_predicated.cpp
        scan/scan_unpacked.cpp
        scan/scan_unpacked_synchronized.cpp
        join/inljoin_unpacked.cpp
        join/inljoin_packed.cpp
        join/inljoin_packed_shared.cpp
        join/inljoin_packed_predicated.cpp
        join/inljoin_packed_predicated_shared.cpp
        join/inljoin_packed_cascade.cpp
        join/inljoin_packed_cascade_shared.cpp
        join/inljoin_packed_cascade_predicated.cpp
        join/inljoin_packed_cascade_predicated_shared.cpp
        join/inljoin_packed_gp_cascade.cpp
        join/inljoin_packed_gp_cascade_shared.cpp
        join/inljoin_packed_gp_cascade_predicated.cpp
        join/inljoin_packed_gp_cascade_predicated_shared.cpp
        join/flat_join.cpp
        join/flat_join_predicated.cpp
        join/intersection.cpp
        join/intersection_predicated.cpp
        join/nway_intersection.cpp
        join/nway_intersection_predicated.cpp
        join/packed_theta_join.cpp
        join/packed_anti_semi_join.cpp
        join/property_filter.cpp
        sink/sink_unpacked.cpp
        sink/sink_packed.cpp
        sink/sink_export.cpp
        ai_operator/ai_base.cpp
        ai_operator/ai_serializer.cpp
        ai_operator/map.cpp
        ai_operator/ftree_reconstructor.cpp
        sink/sink_itr_merged.cpp
        sink/sink_linear.cpp
        sink/sink_min.cpp
        sink/sink_min_itr.cpp
        factorized_ftree/ftree_state_update.cpp
        factorized_ftree/ftree_iterator.cpp
        factorized_ftree/ftree_batch_iterator.cpp
        factorized_ftree/ftree_ancestor_finder.cpp
        vector/bitmask.cpp
        vector/state.cpp
        vector/vector.cpp
        vector/unpacked_vector.cpp
        vector/selection_vector.cpp
        query_variable_to_vector.cpp
        factorized_ftree/factorized_tree.cpp
        factorized_ftree/factorized_tree_element.cpp
        schema/adj_list_manager.cpp
        ai_operator/ftree_reconstructor.cpp)
target_include_directories(liboperator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(liboperator PUBLIC libtable libserder libquery libplan re2::re2)
target_link_libraries(liboperator PUBLIC libflock)

if (STORAGE_TO_VECTOR_MEMCPY_PTR_ALIAS)
    add_compile_definitions(STORAGE_TO_VECTOR_MEMCPY_PTR_ALIAS)
endif ()
if (ENABLE_SELECTION_POS_VECTOR)
    add_compile_definitions(ENABLE_SELECTION_POS_VECTOR)
endif ()