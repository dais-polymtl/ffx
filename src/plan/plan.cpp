#include "include/plan.hpp"
#include "../operator/include/join/flat_join.hpp"
#include "../operator/include/join/flat_join_predicated.hpp"
#include "../operator/include/join/inljoin_packed.hpp"
#include "../operator/include/join/inljoin_packed_cascade.hpp"
#include "../operator/include/join/inljoin_packed_cascade_predicated.hpp"
#include "../operator/include/join/inljoin_packed_cascade_predicated_shared.hpp"
#include "../operator/include/join/inljoin_packed_cascade_shared.hpp"
#include "../operator/include/join/inljoin_packed_gp_cascade.hpp"
#include "../operator/include/join/inljoin_packed_gp_cascade_predicated.hpp"
#include "../operator/include/join/inljoin_packed_gp_cascade_predicated_shared.hpp"
#include "../operator/include/join/inljoin_packed_gp_cascade_shared.hpp"
#include "../operator/include/join/inljoin_packed_predicated.hpp"
#include "../operator/include/join/inljoin_packed_predicated_shared.hpp"
#include "../operator/include/join/inljoin_packed_shared.hpp"
#include "../operator/include/join/inljoin_unpacked.hpp"
#include "../operator/include/join/intersection.hpp"
#include "../operator/include/join/intersection_predicated.hpp"
#include "../operator/include/join/nway_intersection.hpp"
#include "../operator/include/join/nway_intersection_predicated.hpp"
#include "../operator/include/join/packed_anti_semi_join.hpp"
#include "../operator/include/join/packed_theta_join.hpp"
#include "../operator/include/scan/scan.hpp"
#include "../operator/include/scan/scan_predicated.hpp"
#include "../operator/include/scan/scan_synchronized.hpp"
#include "../operator/include/scan/scan_synchronized_predicated.hpp"
#include "../operator/include/scan/scan_unpacked.hpp"
#include "../operator/include/scan/scan_unpacked_synchronized.hpp"
#include "../operator/include/sink/sink_linear.hpp"
#include "../operator/include/sink/sink_min.hpp"
#include "../operator/include/sink/sink_min_itr.hpp"
#include "../operator/include/sink/sink_no_op.hpp"
#include "../operator/include/sink/sink_export.hpp"
#include "../operator/include/ai_operator/map.hpp"
#include "../operator/include/sink/sink_packed.hpp"
#include "../operator/include/sink/sink_unpacked.hpp"

#include <cassert>
#include <iostream>

namespace ffx {


Plan::Plan(std::unique_ptr<Operator> first_operator, const std::vector<std::string>& column_ordering)
    : _first_op(std::move(first_operator)), map(std::make_unique<QueryVariableToVectorMap>()),
      _column_ordering(column_ordering), _predicate_pool(std::make_unique<StringPool>()) {}

inline static void benchmark_barrier() {
    std::atomic_thread_fence(std::memory_order_seq_cst);// prevent hardware reordering
#if defined(__arm__) || defined(__aarch64__)
    // ARM-specific memory barrier if needed
    __asm__ volatile("dmb ish" ::: "memory");
#else
    // x86 and other architectures
    __asm__ volatile("" ::: "memory");
#endif
}

void Plan::init(const std::vector<const Table*>& tables, const std::shared_ptr<FactorizedTreeElement>& root,
                Schema* schema) const {
    schema->map = map.get();
    schema->tables = tables;
    schema->root = root;
    schema->column_ordering = &_column_ordering;
    schema->predicate_pool = _predicate_pool.get();
    _first_op->init(schema);
}

std::chrono::milliseconds Plan::execute() const {
    benchmark_barrier();
    const auto exec_start_time = std::chrono::steady_clock::now();

    _first_op->execute();

    benchmark_barrier();
    const auto exec_end_time = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(exec_end_time - exec_start_time);
}

uint64_t Plan::get_num_output_tuples() const {
    auto curr_op = _first_op->next_op;
    while (curr_op->next_op != nullptr) {
        curr_op = curr_op->next_op;
    }
    return curr_op->get_num_output_tuples();
}

Operator* Plan::get_first_op() const { return _first_op.get(); }

static void add_sink_name(sink_type sink, std::vector<std::string>& operator_names) {
    switch (sink) {
        case SINK_UNPACKED:
            operator_names.emplace_back("SINK_UNPACKED");
            break;
        case SINK_PACKED:
            operator_names.emplace_back("SINK_PACKED");
            break;
        case SINK_MIN:
            operator_names.emplace_back("SINK_MIN");
            break;
        case SINK_PACKED_NOOP:
            operator_names.emplace_back("SINK_NOOP");
            break;
        case SINK_UNPACKED_NOOP:
            operator_names.emplace_back("SINK_NOOP");
            break;
        case SINK_EXPORT_CSV:
            operator_names.emplace_back("SINK_EXPORT_CSV");
            break;
        case SINK_EXPORT_JSON:
            operator_names.emplace_back("SINK_EXPORT_JSON");
            break;
        case SINK_EXPORT_MARKDOWN:
            operator_names.emplace_back("SINK_EXPORT_MARKDOWN");
            break;
        default:
            operator_names.emplace_back("UNKNOWN");
            break;
    }
}

std::vector<std::string> Plan::get_operator_names(sink_type sink) const {
    std::vector<std::string> operator_names;
    Operator* op = _first_op.get();

    // Traverse the operator chain and identify each operator type
    // Note: Check predicated versions BEFORE non-predicated to avoid wrong match
    while (op != nullptr) {
        // Scans (check predicated versions first)
        if (dynamic_cast<ScanUnpackedSynchronized*>(op) != nullptr) {
            operator_names.emplace_back("SCAN_UNPACKED_SYNCHRONIZED");
        } else if (dynamic_cast<ScanUnpacked<>*>(op) != nullptr) {
            operator_names.emplace_back("SCAN_UNPACKED");
        } else if (dynamic_cast<ScanSynchronizedPredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("SCAN_SYNC_PRED");
        } else if (dynamic_cast<ScanPredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("SCAN_PREDICATED");
        } else if (dynamic_cast<ScanSynchronized*>(op) != nullptr) {
            operator_names.emplace_back("SCAN_SYNCHRONIZED");
        } else if (dynamic_cast<Scan<>*>(op) != nullptr) {
            operator_names.emplace_back("SCAN");
        }
        // INL Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (dynamic_cast<INLJoinPackedPredicatedShared<uint64_t>*>(op) != nullptr) {
            operator_names.emplace_back("INLJ_PACKED_PRED_SHARED");
        } else if (dynamic_cast<INLJoinPackedShared<uint64_t>*>(op) != nullptr) {
            operator_names.emplace_back("INLJ_PACKED_SHARED");
        } else if (dynamic_cast<INLJoinPackedPredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("INLJ_PACKED_PRED");
        } else if (dynamic_cast<INLJoinPacked<>*>(op) != nullptr) {
            operator_names.emplace_back("INLJ_PACKED");
        }
        // Cascade Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (dynamic_cast<INLJoinPackedCascadePredicatedShared<uint64_t>*>(op) != nullptr) {
            operator_names.emplace_back("CASCADE_PRED_SHARED");
        } else if (dynamic_cast<INLJoinPackedCascadeShared<uint64_t>*>(op) != nullptr) {
            operator_names.emplace_back("CASCADE_SHARED");
        } else if (dynamic_cast<INLJoinPackedCascadePredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("CASCADE_PRED");
        } else if (dynamic_cast<INLJoinPackedCascade<>*>(op) != nullptr) {
            operator_names.emplace_back("CASCADE");
        }
        // GP Cascade Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (dynamic_cast<INLJoinPackedGPCascadePredicatedShared<uint64_t>*>(op) != nullptr) {
            operator_names.emplace_back("GP_CASCADE_PRED_SHARED");
        } else if (dynamic_cast<INLJoinPackedGPCascadeShared<uint64_t>*>(op) != nullptr) {
            operator_names.emplace_back("GP_CASCADE_SHARED");
        } else if (dynamic_cast<INLJoinPackedGPCascadePredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("GP_CASCADE_PRED");
        } else if (dynamic_cast<INLJoinPackedGPCascade<>*>(op) != nullptr) {
            operator_names.emplace_back("GP_CASCADE");
        }
        // Unpacked Join
        else if (dynamic_cast<INLJoinUnpacked<>*>(op) != nullptr) {
            operator_names.emplace_back("INLJ");
        }
        // Flat Joins (check predicated first)
        else if (dynamic_cast<FlatJoinPredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("FLAT_JOIN_PRED");
        } else if (dynamic_cast<FlatJoin<>*>(op) != nullptr) {
            operator_names.emplace_back("FLAT_JOIN");
        }
        // Intersections (check predicated first)
        else if (dynamic_cast<IntersectionPredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("INTERSECTION_PRED");
        } else if (dynamic_cast<Intersection<>*>(op) != nullptr) {
            operator_names.emplace_back("INTERSECTION");
        }
        // N-Way Intersections (check predicated first)
        else if (dynamic_cast<NWayIntersectionPredicated<>*>(op) != nullptr) {
            operator_names.emplace_back("NWAY_INTERSECTION_PRED");
        } else if (dynamic_cast<NWayIntersection<>*>(op) != nullptr) {
            operator_names.emplace_back("NWAY_INTERSECTION");
        }
        // Theta Join (attribute filter)
        else if (dynamic_cast<PackedThetaJoin<>*>(op) != nullptr) {
            operator_names.emplace_back("THETA_JOIN");
        }
        // Sinks
        else if (dynamic_cast<SinkPacked*>(op) != nullptr) {
            operator_names.emplace_back("SINK_PACKED");
            break;// Sink is always the last operator
        } else if (dynamic_cast<Map*>(op) != nullptr) {
            operator_names.emplace_back("MAP");
        } else if (dynamic_cast<SinkExport*>(op) != nullptr) {
            operator_names.emplace_back("SINK_EXPORT");
            break;
        } else if (dynamic_cast<SinkLinear*>(op) != nullptr) {
            operator_names.emplace_back("SINK_LINEAR");
            break;// Sink is always the last operator
        } else if (dynamic_cast<SinkUnpacked*>(op) != nullptr) {
            operator_names.emplace_back("SINK_UNPACKED");
            break;// Sink is always the last operator
        } else if (dynamic_cast<SinkMin*>(op) != nullptr || dynamic_cast<SinkMinItr*>(op) != nullptr) {
            operator_names.emplace_back("SINK_MIN");
            break;// Sink is always the last operator
        } else if (dynamic_cast<SinkNoOp*>(op) != nullptr) {
            operator_names.emplace_back("SINK_NOOP");
            break;// Sink is always the last operator
        } else {
            operator_names.emplace_back("UNKNOWN");
        }
        op = op->next_op;
    }

    return operator_names;
}

void Plan::print_pipeline() const {
    std::cout << "=== Operator Pipeline ===" << std::endl;

    Operator* op = _first_op.get();
    int idx = 0;

    while (op != nullptr) {
        std::string op_name;
        std::string details;

        // Scans (check predicated first)
        if (auto* scan_up_sync = dynamic_cast<ScanUnpackedSynchronized*>(op)) {
            op_name = "ScanUnpackedSynchronized";
            details = "attr=" + scan_up_sync->attribute();
        } else if (auto* scan_up = dynamic_cast<ScanUnpacked<>*>(op)) {
            op_name = "ScanUnpacked";
            details = "attr=" + scan_up->attribute();
        } else if (auto* scan_sync_pred = dynamic_cast<ScanSynchronizedPredicated<>*>(op)) {
            op_name = "ScanSynchronizedPredicated";
            details = "attr=" + scan_sync_pred->attribute();
            if (scan_sync_pred->has_predicate()) { details += ", pred=" + scan_sync_pred->predicate_string(); }
        } else if (auto* scan_pred = dynamic_cast<ScanPredicated<>*>(op)) {
            op_name = "ScanPredicated";
            details = "attr=" + scan_pred->attribute();
            if (scan_pred->has_predicate()) { details += ", pred=[" + scan_pred->predicate_string() + "]"; }
        } else if (auto* scan_sync = dynamic_cast<ScanSynchronized*>(op)) {
            op_name = "ScanSynchronized";
            details = "attr=" + scan_sync->attribute();
        } else if (auto* scan = dynamic_cast<Scan<>*>(op)) {
            op_name = "Scan";
            details = "attr=" + scan->attribute();
        }
        // INL Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (auto* join_pred_shared = dynamic_cast<INLJoinPackedPredicatedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedPredicatedShared";
            details = join_pred_shared->join_key() + "->" + join_pred_shared->output_key();
        } else if (auto* join_shared = dynamic_cast<INLJoinPackedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedShared";
            details = join_shared->join_key() + "->" + join_shared->output_key();
        } else if (auto* join_pred = dynamic_cast<INLJoinPackedPredicated<>*>(op)) {
            op_name = "INLJoinPackedPredicated";
            details = join_pred->join_key() + "->" + join_pred->output_key();
            if (join_pred->has_predicate()) { details += ", pred=[" + join_pred->predicate_string() + "]"; }
        } else if (auto* join = dynamic_cast<INLJoinPacked<>*>(op)) {
            op_name = "INLJoinPacked";
            details = join->join_key() + "->" + join->output_key();
        }
        // Cascade Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (auto* casc_pred_shared = dynamic_cast<INLJoinPackedCascadePredicatedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedCascadePredicatedShared";
            details = casc_pred_shared->join_key() + "->" + casc_pred_shared->output_key();
        } else if (auto* casc_shared = dynamic_cast<INLJoinPackedCascadeShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedCascadeShared";
            details = casc_shared->join_key() + "->" + casc_shared->output_key();
        } else if (auto* casc_pred = dynamic_cast<INLJoinPackedCascadePredicated<>*>(op)) {
            op_name = "INLJoinPackedCascadePredicated";
            details = casc_pred->join_key() + "->" + casc_pred->output_key();
            if (casc_pred->has_predicate()) { details += ", pred=[" + casc_pred->predicate_string() + "]"; }
        } else if (auto* casc = dynamic_cast<INLJoinPackedCascade<>*>(op)) {
            op_name = "INLJoinPackedCascade";
            details = casc->join_key() + "->" + casc->output_key();
        }
        // GP Cascade Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (auto* gp_pred_shared = dynamic_cast<INLJoinPackedGPCascadePredicatedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedGPCascadePredicatedShared";
            details = gp_pred_shared->join_key() + "->" + gp_pred_shared->output_key();
        } else if (auto* gp_shared = dynamic_cast<INLJoinPackedGPCascadeShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedGPCascadeShared";
            details = gp_shared->join_key() + "->" + gp_shared->output_key();
        } else if (auto* gp_pred = dynamic_cast<INLJoinPackedGPCascadePredicated<>*>(op)) {
            op_name = "INLJoinPackedGPCascadePredicated";
            details = gp_pred->join_key() + "->" + gp_pred->output_key();
            if (gp_pred->has_predicate()) { details += ", pred=[" + gp_pred->predicate_string() + "]"; }
        } else if (auto* gp = dynamic_cast<INLJoinPackedGPCascade<>*>(op)) {
            op_name = "INLJoinPackedGPCascade";
            details = gp->join_key() + "->" + gp->output_key();
        }
        // Unpacked Join
        else if (auto* unpack = dynamic_cast<INLJoinUnpacked<>*>(op)) {
            op_name = "INLJoinUnpacked";
            details = unpack->join_key() + "->" + unpack->output_key();
        }
        // Flat Joins (check predicated first)
        else if (auto* flat_pred = dynamic_cast<FlatJoinPredicated<>*>(op)) {
            op_name = "FlatJoinPredicated";
            details = "parent=" + flat_pred->parent_attr() + ", lca=" + flat_pred->lca_attr() +
                      ", out=" + flat_pred->output_attr();
            if (flat_pred->has_predicate()) { details += ", pred=[" + flat_pred->predicate_string() + "]"; }
        } else if (auto* flat = dynamic_cast<FlatJoin<>*>(op)) {
            op_name = "FlatJoin";
            details = "parent=" + flat->parent_attr() + ", lca=" + flat->lca_attr() + ", out=" + flat->output_attr();
        }
        // Intersections (check predicated first)
        else if (auto* inter_pred = dynamic_cast<IntersectionPredicated<>*>(op)) {
            op_name = "IntersectionPredicated";
            details = "anc=" + inter_pred->ancestor_attr() + ", desc=" + inter_pred->descendant_attr() +
                      ", out=" + inter_pred->output_attr();
            if (inter_pred->has_predicate()) { details += ", pred=[" + inter_pred->predicate_string() + "]"; }
        } else if (auto* inter = dynamic_cast<Intersection<>*>(op)) {
            op_name = "Intersection";
            details = "anc=" + inter->ancestor_attr() + ", desc=" + inter->descendant_attr() +
                      ", out=" + inter->output_attr();
        }
        // N-Way Intersections (check predicated first)
        else if (auto* nway_pred = dynamic_cast<NWayIntersectionPredicated<>*>(op)) {
            op_name = "NWayIntersectionPredicated";
            details = "out=" + nway_pred->output_attr();
            if (nway_pred->has_predicate()) { details += ", pred=[" + nway_pred->predicate_string() + "]"; }
        } else if (auto* nway = dynamic_cast<NWayIntersection<>*>(op)) {
            op_name = "NWayIntersection";
            details = "out=" + nway->output_attr();
        }
        // Theta Join
        else if (auto* theta = dynamic_cast<PackedThetaJoin<>*>(op)) {
            op_name = "PackedThetaJoin";
            details = theta->left_attr() + " " + predicate_op_to_string(theta->op()) + " " + theta->right_attr();
        }
        // Anti-Semi-Join
        else if (auto* anti = dynamic_cast<PackedAntiSemiJoin<>*>(op)) {
            op_name = "PackedAntiSemiJoin";
            details = "NOT " + anti->left_attr() + "->" + anti->right_attr();
        }
        // Sinks
        else if (dynamic_cast<SinkPacked*>(op) != nullptr) {
            op_name = "SinkPacked";
        } else if (dynamic_cast<Map*>(op) != nullptr) {
            op_name = "Map";
        } else if (dynamic_cast<SinkExport*>(op) != nullptr) {
            op_name = "SinkExport";
        } else if (dynamic_cast<SinkLinear*>(op) != nullptr) {
            op_name = "SinkLinear";
        } else if (dynamic_cast<SinkUnpacked*>(op) != nullptr) {
            op_name = "SinkUnpacked";
        } else if (dynamic_cast<SinkMin*>(op) != nullptr || dynamic_cast<SinkMinItr*>(op) != nullptr) {
            op_name = "SinkMin";
        } else if (dynamic_cast<SinkNoOp*>(op) != nullptr) {
            op_name = "SinkNoOp";
        } else {
            op_name = "Unknown";
        }

        std::cout << "  [" << idx << "] " << op_name;
        if (!details.empty()) { std::cout << " (" << details << ")"; }
        std::cout << std::endl;

        op = op->next_op;
        idx++;
    }

    std::cout << "=========================" << std::endl;
}

void print_operator_chain(Operator* first_op) {
    std::cout << "=== Operator Pipeline ===" << std::endl;

    Operator* op = first_op;
    int idx = 0;

    while (op != nullptr) {
        std::string op_name;
        std::string details;

        // Scans (check predicated versions first)
        if (auto* scan_up_sync = dynamic_cast<ScanUnpackedSynchronized*>(op)) {
            op_name = "ScanUnpackedSynchronized";
            details = "attr=" + scan_up_sync->attribute();
        } else if (auto* scan_up = dynamic_cast<ScanUnpacked<>*>(op)) {
            op_name = "ScanUnpacked";
            details = "attr=" + scan_up->attribute();
        } else if (auto* scan_sync_pred = dynamic_cast<ScanSynchronizedPredicated<>*>(op)) {
            op_name = "ScanSynchronizedPredicated";
            details = "attr=" + scan_sync_pred->attribute();
            if (scan_sync_pred->has_predicate()) { details += ", pred=" + scan_sync_pred->predicate_string(); }
        } else if (auto* scan_pred = dynamic_cast<ScanPredicated<>*>(op)) {
            op_name = "ScanPredicated";
            details = "attr=" + scan_pred->attribute();
            if (scan_pred->has_predicate()) { details += ", pred=" + scan_pred->predicate_string(); }
        } else if (auto* scan_sync = dynamic_cast<ScanSynchronized*>(op)) {
            op_name = "ScanSynchronized";
            details = "attr=" + scan_sync->attribute();
        } else if (auto* scan = dynamic_cast<Scan<>*>(op)) {
            op_name = "Scan";
            details = "attr=" + scan->attribute();
        }
        // INL Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (auto* join_pred_shared = dynamic_cast<INLJoinPackedPredicatedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedPredicatedShared";
            details = join_pred_shared->join_key() + "->" + join_pred_shared->output_key();
        } else if (auto* join_shared = dynamic_cast<INLJoinPackedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedShared";
            details = join_shared->join_key() + "->" + join_shared->output_key();
        } else if (auto* join_pred = dynamic_cast<INLJoinPackedPredicated<>*>(op)) {
            op_name = "INLJoinPackedPredicated";
            details = join_pred->join_key() + "->" + join_pred->output_key();
        } else if (auto* join = dynamic_cast<INLJoinPacked<>*>(op)) {
            op_name = "INLJoinPacked";
            details = join->join_key() + "->" + join->output_key();
        }
        // Cascade Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (auto* casc_pred_shared = dynamic_cast<INLJoinPackedCascadePredicatedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedCascadePredicatedShared";
            details = casc_pred_shared->join_key() + "->" + casc_pred_shared->output_key();
        } else if (auto* casc_shared = dynamic_cast<INLJoinPackedCascadeShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedCascadeShared";
            details = casc_shared->join_key() + "->" + casc_shared->output_key();
        } else if (auto* casc_pred = dynamic_cast<INLJoinPackedCascadePredicated<>*>(op)) {
            op_name = "INLJoinPackedCascadePredicated";
            details = casc_pred->join_key() + "->" + casc_pred->output_key();
        } else if (auto* casc = dynamic_cast<INLJoinPackedCascade<>*>(op)) {
            op_name = "INLJoinPackedCascade";
            details = casc->join_key() + "->" + casc->output_key();
        }
        // GP Cascade Joins (check shared predicated first, then regular predicated, then shared, then regular)
        else if (auto* gp_pred_shared = dynamic_cast<INLJoinPackedGPCascadePredicatedShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedGPCascadePredicatedShared";
            details = gp_pred_shared->join_key() + "->" + gp_pred_shared->output_key();
        } else if (auto* gp_shared = dynamic_cast<INLJoinPackedGPCascadeShared<uint64_t>*>(op)) {
            op_name = "INLJoinPackedGPCascadeShared";
            details = gp_shared->join_key() + "->" + gp_shared->output_key();
        } else if (auto* gp_pred = dynamic_cast<INLJoinPackedGPCascadePredicated<>*>(op)) {
            op_name = "INLJoinPackedGPCascadePredicated";
            details = gp_pred->join_key() + "->" + gp_pred->output_key();
        } else if (auto* gp = dynamic_cast<INLJoinPackedGPCascade<>*>(op)) {
            op_name = "INLJoinPackedGPCascade";
            details = gp->join_key() + "->" + gp->output_key();
        }
        // Unpacked Join
        else if (auto* unpack = dynamic_cast<INLJoinUnpacked<>*>(op)) {
            op_name = "INLJoinUnpacked";
            details = unpack->join_key() + "->" + unpack->output_key();
        }
        // Flat Joins (check predicated first)
        else if (auto* flat_pred = dynamic_cast<FlatJoinPredicated<>*>(op)) {
            op_name = "FlatJoinPredicated";
            details = "parent=" + flat_pred->parent_attr() + ", lca=" + flat_pred->lca_attr() +
                      ", out=" + flat_pred->output_attr();
        } else if (auto* flat = dynamic_cast<FlatJoin<>*>(op)) {
            op_name = "FlatJoin";
            details = "parent=" + flat->parent_attr() + ", lca=" + flat->lca_attr() + ", out=" + flat->output_attr();
        }
        // Intersections (check predicated first)
        else if (auto* inter_pred = dynamic_cast<IntersectionPredicated<>*>(op)) {
            op_name = "IntersectionPredicated";
            details = "anc=" + inter_pred->ancestor_attr() + ", desc=" + inter_pred->descendant_attr() +
                      ", out=" + inter_pred->output_attr();
        } else if (auto* inter = dynamic_cast<Intersection<>*>(op)) {
            op_name = "Intersection";
            details = "anc=" + inter->ancestor_attr() + ", desc=" + inter->descendant_attr() +
                      ", out=" + inter->output_attr();
        }
        // N-Way Intersections (check predicated first)
        else if (auto* nway_pred = dynamic_cast<NWayIntersectionPredicated<>*>(op)) {
            op_name = "NWayIntersectionPredicated";
            details = "out=" + nway_pred->output_attr();
        } else if (auto* nway = dynamic_cast<NWayIntersection<>*>(op)) {
            op_name = "NWayIntersection";
            details = "out=" + nway->output_attr();
        }
        // Theta Join
        else if (auto* theta = dynamic_cast<PackedThetaJoin<>*>(op)) {
            op_name = "PackedThetaJoin";
            details = theta->left_attr() + " <op> " + theta->right_attr();
        }
        // Sinks
        else if (dynamic_cast<SinkPacked*>(op) != nullptr) {
            op_name = "SinkPacked";
        } else if (dynamic_cast<SinkLinear*>(op) != nullptr) {
            op_name = "SinkLinear";
        } else if (dynamic_cast<SinkUnpacked*>(op) != nullptr) {
            op_name = "SinkUnpacked";
        } else if (dynamic_cast<SinkMin*>(op) != nullptr || dynamic_cast<SinkMinItr*>(op) != nullptr) {
            op_name = "SinkMin";
        } else if (dynamic_cast<SinkNoOp*>(op) != nullptr) {
            op_name = "SinkNoOp";
        } else {
            op_name = "Unknown";
        }

        std::cout << "  [" << idx << "] " << op_name;
        if (!details.empty()) { std::cout << " (" << details << ")"; }
        std::cout << std::endl;

        op = op->next_op;
        idx++;
    }

    std::cout << "=========================" << std::endl;
}

}// namespace ffx
