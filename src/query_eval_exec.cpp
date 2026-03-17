#include "benchmark/include/benchmark.hpp"
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                        << " <serialized_root_dir> <query> <column_ordering> "
                            "[sink_type] [-mode <1|0>] [-t <num_threads>]"
                  << std::endl;
        std::cerr << std::endl;
        std::cerr << "Query string: use Q(head) := body (optional WHERE ...)." << std::endl;
        std::cerr << "  Projection: \"Q(a,b,c) := R(a,b),R(b,c)\"" << std::endl;
        std::cerr << "  With WHERE: \"Q(a,b,c) := R(a,b),R(b,c) WHERE EQ(a,5) AND GTE(b,10)\"" << std::endl;
        std::cerr << "  Aggregates / LLM:" << std::endl;
        std::cerr << "    Q(MIN(a,b)) := R(a,b), R(b,c)     — use sink 'min'; MIN(...) lists attrs to aggregate" << std::endl;
        std::cerr << "    Q(COUNT(*)) := R(a,b), R(b,c)      — tuple count with packed / unpacked sinks" << std::endl;
        std::cerr << "    Q(NOOP) := R(a,b)" << std::endl;
        std::cerr << "    Q(a,b,llm) := R(a,b), llm = LLM_MAP({\"model\":\"...\"})" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Supported predicates:" << std::endl;
        std::cerr << "  EQ(attr,value)   - Equal" << std::endl;
        std::cerr << "  NEQ(attr,value)  - Not equal" << std::endl;
        std::cerr << "  LT(attr,value)   - Less than" << std::endl;
        std::cerr << "  GT(attr,value)   - Greater than" << std::endl;
        std::cerr << "  LTE(attr,value)  - Less than or equal" << std::endl;
        std::cerr << "  GTE(attr,value)  - Greater than or equal" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Logical operators: AND, OR (with parentheses for grouping)" << std::endl;
        std::cerr << "Optional mode: -mode 1 (packed, default) | -mode 0 (unpacked)" << std::endl;
        std::cerr << "Optional threading: -t <num_threads>" << std::endl;
        return 1;
    }

    try {

    std::string sink_type = "packed";
    uint32_t num_threads = 1;
    bool mode_explicitly_set = false;

    std::vector<std::string> passthrough_args;
    passthrough_args.reserve(argc > 4 ? static_cast<size_t>(argc - 4) : 0);
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-t") {
            if (i + 1 >= argc) { throw std::invalid_argument("Missing value for -t"); }
            const std::string thread_arg = argv[++i];
            int parsed_threads = std::stoi(thread_arg);
            if (parsed_threads <= 0) { throw std::invalid_argument("-t must be >= 1"); }
            num_threads = static_cast<uint32_t>(parsed_threads);
            continue;
        }
        if (arg == "-mode") {
            if (i + 1 >= argc) { throw std::invalid_argument("Missing value for -mode"); }
            const std::string mode_arg = argv[++i];
            if (mode_arg == "1") {
                sink_type = "packed";
            } else if (mode_arg == "0") {
                sink_type = "unpacked";
            } else {
                throw std::invalid_argument("-mode must be 1 (packed) or 0 (unpacked)");
            }
            mode_explicitly_set = true;
            continue;
        }
        passthrough_args.push_back(arg);
    }

    auto is_sink_token = [](const std::string& token) {
        // Core sinks
        if (token == "packed" || token == "unpacked" || token == "pnoop" || token == "upnoop" ||
            token == "cascade" || token == "min") {
            return true;
        }
        // Export sinks (optionally include an output path suffix, e.g. sink_csv:/tmp/out.csv)
        if (token == "sink_csv" || token.rfind("sink_csv:", 0) == 0) return true;
        if (token == "sink_json" || token.rfind("sink_json:", 0) == 0) return true;
        if (token == "sink_markdown" || token.rfind("sink_markdown:", 0) == 0) return true;
        return false;
    };

    if (!passthrough_args.empty() && is_sink_token(passthrough_args[0])) {
        sink_type = passthrough_args[0];
        passthrough_args.erase(passthrough_args.begin());
    }

    if (mode_explicitly_set &&
        !(sink_type == "packed" || sink_type == "unpacked" || sink_type == "pnoop" || sink_type == "upnoop")) {
        throw std::invalid_argument("-mode can only be used with packed/unpacked sink variants");
    }

    std::vector<std::string> parsed_values;

    if (!passthrough_args.empty()) {
        throw std::invalid_argument(
                "Unexpected arguments after sink_type. For min sink, list attributes in the query head: "
                "Q(MIN(a,b,...)) := ... Optional flags: -mode, -t.");
    }

    // Query must use rule syntax, e.g. Q(a,b,c) := R(a,b),R(b,c) WHERE EQ(a,5) AND GTE(b,10)
    if (num_threads <= 1) {
        ffx::evaluate_query(
            /* serialized_root_dir */ std::string(argv[1]),
                /* query */ std::string(argv[2]),
                /* column_ordering */ std::string(argv[3]),
                /* sink_type */ sink_type,
                /* expected_values */ parsed_values);
    } else {
        ffx::evaluate_query_multithreaded(
            /* serialized_root_dir */ std::string(argv[1]),
                /* query */ std::string(argv[2]),
                /* column_ordering */ std::string(argv[3]),
                /* sink_type */ sink_type,
                /* expected_values */ parsed_values,
                /* num_threads */ num_threads);
    }

    } catch (const std::exception& e) {
        std::cerr << "query_eval_exec: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "query_eval_exec: unknown exception" << std::endl;
        return 2;
    }

    return 0;
}
