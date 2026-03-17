#pragma once

// Lightweight metrics manager for FFX (no DuckDB dependency).
//
// This header-only implementation aggregates basic LLM usage statistics
// across the lifetime of the process. Providers update these counters via
// MetricsManager::UpdateTokens / IncrementApiCalls / AddApiDuration, and
// sinks (e.g., SinkExport) can persist them alongside query outputs by
// calling WriteToFile().

#include <atomic>
#include <cstdint>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace flock {

class MetricsManager {
public:
    // Accumulate token usage for a batch of requests.
    static void UpdateTokens(int64_t input, int64_t output) {
        if (input > 0) {
            _input_tokens.fetch_add(input, std::memory_order_relaxed);
        }
        if (output > 0) {
            _output_tokens.fetch_add(output, std::memory_order_relaxed);
        }
    }

    // Increment the number of API calls by one.
    static void IncrementApiCalls() { _api_calls.fetch_add(1, std::memory_order_relaxed); }

    // Add API duration (in milliseconds) to the running total.
    static void AddApiDuration(double duration_ms) {
        if (duration_ms > 0.0) {
            auto delta = static_cast<int64_t>(duration_ms);
            _api_duration_ms.fetch_add(delta, std::memory_order_relaxed);
        }
    }

    // Reset all counters to zero (intended to be called at the start of a query).
    static void Reset() {
        _input_tokens.store(0, std::memory_order_relaxed);
        _output_tokens.store(0, std::memory_order_relaxed);
        _api_calls.store(0, std::memory_order_relaxed);
        _api_duration_ms.store(0, std::memory_order_relaxed);
    }

    // Snapshot the current metrics as a JSON object.
    static nlohmann::json DumpToJson() {
        const auto in = _input_tokens.load(std::memory_order_relaxed);
        const auto out = _output_tokens.load(std::memory_order_relaxed);
        const auto total = in + out;
        const auto calls = _api_calls.load(std::memory_order_relaxed);
        const auto dur_ms = _api_duration_ms.load(std::memory_order_relaxed);

        nlohmann::json j;
        j["input_tokens"] = in;
        j["output_tokens"] = out;
        j["total_tokens"] = total;
        j["api_calls"] = calls;
        j["api_duration_ms"] = dur_ms;
        return j;
    }

    // Write metrics to a JSON file. On failure, this is a no-op.
    static void WriteToFile(const std::string& path) {
        try {
            std::ofstream out(path, std::ios::out | std::ios::trunc);
            if (!out.is_open()) {
                return;
            }
            const auto j = DumpToJson();
            out << j.dump(2);
            out.close();
        } catch (...) {
            // Swallow all exceptions – metrics should never break query execution.
        }
    }

private:
    static std::atomic<int64_t> _input_tokens;
    static std::atomic<int64_t> _output_tokens;
    static std::atomic<int64_t> _api_calls;
    static std::atomic<int64_t> _api_duration_ms;
};

// Static member definitions.
inline std::atomic<int64_t> MetricsManager::_input_tokens{0};
inline std::atomic<int64_t> MetricsManager::_output_tokens{0};
inline std::atomic<int64_t> MetricsManager::_api_calls{0};
inline std::atomic<int64_t> MetricsManager::_api_duration_ms{0};

} // namespace flock

