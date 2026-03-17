#include "sink/sink_min_itr.hpp"
#include "schema/schema.hpp"
#include "../../table/include/table.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace ffx {

void SinkMinItr::init(Schema* schema) {
    _iterator = std::make_unique<FTreeIterator>();
    _iterator->init(schema);

    _num_output_tuples = 0;

    // Get min_values array and size from schema
    assert(schema);
    assert(schema->min_values);
    assert(schema->min_values_size > 0);

    _min_values = schema->min_values;
    _min_values_size = schema->min_values_size;

    // Initialize all min values to maximum possible value
    for (size_t i = 0; i < _min_values_size; i++) {
        _min_values[i] = std::numeric_limits<uint64_t>::max();
    }

    // Initialize string attribute support
    _dictionary = schema->dictionary;

    if (schema->required_min_attrs.empty()) {
        throw std::runtime_error(
                "SinkMinItr: schema.required_min_attrs must be non-empty; use Q(MIN(a,b,...)) := ... with the min "
                "sink");
    }
    const std::vector<std::string>& target_attrs = schema->required_min_attrs;

    // Map target attributes to their indices in the column ordering
    // Attributes not in column_ordering are resolved as property attributes via adj_list
    if (schema->column_ordering) {
        const auto& col_ordering = *schema->column_ordering;

        for (const auto& attr: target_attrs) {
            // Try to find in column ordering first
            bool found_in_ordering = false;
            for (size_t i = 0; i < col_ordering.size(); ++i) {
                if (col_ordering[i] == attr) {
                    _mapped_indices.push_back(i);
                    _is_string_attr.push_back(schema->string_attributes && schema->string_attributes->count(attr) > 0);
                    _min_attr_type.push_back(-1);// direct column
                    found_in_ordering = true;
                    break;
                }
            }

            if (!found_in_ordering) {
                // Property attribute — find entity attr and adj_list via table lookup
                bool resolved = false;

                // First try schema adj_list_map
                for (size_t ci = 0; ci < col_ordering.size(); ++ci) {
                    const auto& entity = col_ordering[ci];
                    if (schema->has_adj_list(entity, attr)) {
                        PropertyMinAttr pma;
                        pma.entity_col_idx = ci;
                        pma.adj_lists = schema->get_adj_list(entity, attr);
                        pma.is_string = (schema->string_attributes && schema->string_attributes->count(attr) > 0);

                        _min_attr_type.push_back(static_cast<int>(_property_min_attrs.size()));
                        _mapped_indices.push_back(ci);// entity column index (used as placeholder)
                        _is_string_attr.push_back(pma.is_string);
                        _property_min_attrs.push_back(pma);
                        _has_property_attrs = true;
                        resolved = true;
                        break;
                    }
                }

                if (!resolved) {
                    // Fall back to table scan
                    for (const auto* table: schema->tables) {
                        int entity_idx = -1, prop_idx = -1;
                        for (size_t c = 0; c < table->columns.size(); ++c) {
                            if (table->columns[c] == attr) prop_idx = static_cast<int>(c);
                        }
                        if (prop_idx == -1) continue;

                        for (size_t c = 0; c < table->columns.size(); ++c) {
                            for (size_t ci = 0; ci < col_ordering.size(); ++ci) {
                                if (table->columns[c] == col_ordering[ci]) {
                                    entity_idx = static_cast<int>(c);
                                    PropertyMinAttr pma;
                                    pma.entity_col_idx = ci;
                                    bool is_fwd = (entity_idx < prop_idx);
                                    pma.adj_lists = reinterpret_cast<AdjList<uint64_t>*>(
                                            is_fwd ? table->fwd_adj_lists : table->bwd_adj_lists);
                                    pma.is_string = (schema->string_attributes &&
                                                     schema->string_attributes->count(attr) > 0);

                                    _min_attr_type.push_back(static_cast<int>(_property_min_attrs.size()));
                                    _mapped_indices.push_back(ci);
                                    _is_string_attr.push_back(pma.is_string);
                                    _property_min_attrs.push_back(pma);
                                    _has_property_attrs = true;
                                    resolved = true;
                                    break;
                                }
                            }
                            if (resolved) break;
                        }
                        if (resolved) break;
                    }
                }

                if (!resolved) {
                    throw std::runtime_error("SinkMinItr: Required attribute " + attr +
                                             " not found in column ordering or as property attribute");
                }
            }
        }
    }

    // Set string support on the iterator - note: iterator still uses full column ordering
    // but the sink will only use mapped indices
    std::vector<bool> full_is_string_attr;
    if (schema->column_ordering) {
        for (const auto& attr: *schema->column_ordering) {
            full_is_string_attr.push_back(schema->string_attributes && schema->string_attributes->count(attr) > 0);
        }
    }
    _iterator->set_string_support(_dictionary, full_is_string_attr);

    // Ensure we have the expected number of attributes
    assert(_mapped_indices.size() == _min_values_size);

    // Pre-allocate buffer
    _tuple_size = _iterator->tuple_size();
}

void SinkMinItr::execute() {
    num_exec_call++;

    // Initialize the iterator
    _iterator->initialize_iterators();
    auto num_output_tuples = 0;

    // We need a temporary buffer that matches the FULL iterator tuple size
    auto* full_min_values = static_cast<uint64_t*>(alloca(_tuple_size * sizeof(uint64_t)));
    std::fill_n(full_min_values, _tuple_size, std::numeric_limits<uint64_t>::max());

    auto* tuple_buffer = static_cast<uint64_t*>(alloca(_tuple_size * sizeof(uint64_t)));

    if (_has_property_attrs) {
        // Property attribute path: enumerate all tuples, look up property values
        // next() reads current tuple into buffer THEN advances; returns false when exhausted
        // but the buffer still contains the last valid tuple, so always process after each call.
        if (_iterator->is_valid()) {
            bool has_more;
            do {
                has_more = _iterator->next(tuple_buffer);
                num_output_tuples++;
                for (size_t i = 0; i < _min_values_size; i++) {
                    uint64_t val;
                    if (_min_attr_type[i] < 0) {
                        val = tuple_buffer[_mapped_indices[i]];
                    } else {
                        const auto& pma = _property_min_attrs[_min_attr_type[i]];
                        uint64_t entity_id = tuple_buffer[pma.entity_col_idx];
                        const auto& adj = pma.adj_lists[entity_id];
                        val = (adj.size > 0) ? adj.values[0] : std::numeric_limits<uint64_t>::max();
                    }
                    if (_is_string_attr[i] && _dictionary) {
                        if (_min_values[i] == std::numeric_limits<uint64_t>::max()) {
                            _min_values[i] = val;
                        } else if (val != std::numeric_limits<uint64_t>::max()) {
                            auto& str_val = _dictionary->get_string(val);
                            auto& str_min = _dictionary->get_string(_min_values[i]);
                            if (str_val < str_min) { _min_values[i] = val; }
                        }
                    } else {
                        if (val < _min_values[i]) { _min_values[i] = val; }
                    }
                }
            } while (has_more);
        }
    } else {
        // Original optimized path: no property attributes
        // Map existing mins into the full buffer
        for (size_t i = 0; i < _mapped_indices.size(); i++) {
            full_min_values[_mapped_indices[i]] = _min_values[i];
        }

        // next_min_tuple now processes current tuple before advancing
        while (_iterator->next_min_tuple(full_min_values)) {
            num_output_tuples++;
        }

        // Map final mins back to sink's array
        for (size_t i = 0; i < _mapped_indices.size(); i++) {
            _min_values[i] = full_min_values[_mapped_indices[i]];
        }
    }

    _num_output_tuples += num_output_tuples;
}

}// namespace ffx
