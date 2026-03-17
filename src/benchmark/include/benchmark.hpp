#ifndef VFENGINE_BENCHMARK_HH
#define VFENGINE_BENCHMARK_HH

#include <cstdint>
#include <string>
#include <vector>

namespace ffx {
bool evaluate_query(const std::string& ser_directory, const std::string& query_as_str,
                    const std::string& column_ordering, const std::string& sink_type,
                    const std::vector<std::string>& expected_values);

bool evaluate_query_multithreaded(const std::string& ser_directory, const std::string& query_as_str,
                                  const std::string& column_ordering, const std::string& sink_type,
                                  const std::vector<std::string>& expected_values, std::uint32_t num_threads = 0);
}// namespace ffx

#endif
