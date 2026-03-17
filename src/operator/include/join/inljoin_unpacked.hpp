#ifndef VFENGINE_INDEX_NESTED_LOOP_JOIN_OPERATOR_HH
#define VFENGINE_INDEX_NESTED_LOOP_JOIN_OPERATOR_HH

#include "../../table/include/ffx_str_t.hpp"
#include "operator.hpp"
#include "vector/unpacked_vector.hpp"
#include <string>
#include <type_traits>

namespace ffx {

template<typename T = uint64_t>
class INLJoinUnpacked final : public Operator {
public:
    INLJoinUnpacked() = delete;
    INLJoinUnpacked(const INLJoinUnpacked&) = delete;
    INLJoinUnpacked(std::string join_key, std::string output_key, bool /*is_join_index_fwd*/)
        : Operator(), _join_key(std::move(join_key)), _output_key(std::move(output_key)) {}

    void init(Schema* schema) override;
    void execute() override;

    // Getters for testing/debugging
    const std::string& join_key() const { return _join_key; }
    const std::string& output_key() const { return _output_key; }

private:
    void loop_and_process_join_keys();
    void process_join_key();

    const std::string _join_key, _output_key;
    UnpackedVector<T>* _join_key_vector{};
    UnpackedVector<T>* _output_key_vector{};
    AdjList<T>* _adj_lists;
};

// Type aliases for convenience
using INLJoinUnpackedUint64 = INLJoinUnpacked<uint64_t>;
using INLJoinUnpackedString = INLJoinUnpacked<ffx_str_t>;

}// namespace ffx

#endif
