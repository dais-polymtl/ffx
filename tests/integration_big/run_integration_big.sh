#!/bin/bash
set -e

# Arguments: <build_dir> <test_src_dir>
BUILD_DIR=$1
TEST_SRC_DIR=$2

SERIALIZER="${BUILD_DIR}/table_serializer"
TEST_EXE="${BUILD_DIR}/integration_big_test"
GEN_SCRIPT="${TEST_SRC_DIR}/generate_data.py"

# Use a temporary directory for serialized data
DATA_DIR=$(mktemp -d /tmp/ffx_integration_big_XXXXXX)
trap 'rm -rf "$DATA_DIR"' EXIT

echo "Using temporary data directory: ${DATA_DIR}"

# 1. Generate Data
python3 "${GEN_SCRIPT}" "${DATA_DIR}"

# 2. Serialize
mkdir -p "${DATA_DIR}/serialized/papers" "${DATA_DIR}/serialized/edges"
"${SERIALIZER}" "${DATA_DIR}/papers.csv" "${DATA_DIR}/serialized/papers" "${TEST_SRC_DIR}/ser_papers_config.txt"
"${SERIALIZER}" "${DATA_DIR}/edges.csv" "${DATA_DIR}/serialized/edges" "${TEST_SRC_DIR}/ser_edges_config.txt"

# 3. Run Tests
"${TEST_EXE}" --data_dir="${DATA_DIR}/serialized"

for POS in pos2 pos3 pos4 pos5 end; do
    EXE="${BUILD_DIR}/integration_big_llm_${POS}_test"
    if [ -x "${EXE}" ]; then
        echo "Running LLM ${POS} test..."
        "${EXE}" --data_dir="${DATA_DIR}/serialized"
    fi
done

echo "All tests completed successfully."
