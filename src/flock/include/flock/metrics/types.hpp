#pragma once

#include <cstdint>

namespace flock {

enum class FunctionType : uint8_t {
    LLM_MAP = 0,
    LLM_FILTER = 1,
    LLM_EMBEDDING = 2,
    LLM_REDUCE = 3,
    LLM_RERANK = 4,
    LLM_FIRST = 5,
    LLM_LAST = 6,
    UNKNOWN = 7
};

}// namespace flock
