#include "sink/sink_unpacked.hpp"

namespace ffx {

void SinkUnpacked::init(Schema* schema) {
    auto& map = *schema->map;

    _num_output_tuples = 0;
    _num_list_vectors = map.get_num_unpacked_list_vectors();
    _states_of_list_vectors = std::make_unique<UnpackedState*[]>(_num_list_vectors);
    map.get_unpacked_states_of_list_vectors(_states_of_list_vectors.get());
}

void SinkUnpacked::execute() {
    num_exec_call++;
    uint64_t curr{1};
    for (size_t i = 0; i < _num_list_vectors; i++) {
        curr *= _states_of_list_vectors[i]->size;
    }
    _num_output_tuples += curr;
}

}// namespace ffx
