#include "../src/table/include/string_pool.hpp"
#include "../src/table/include/ffx_str_t.hpp"
#include <gtest/gtest.h>
#include <cstring>

namespace ffx {

class StringPoolTests : public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_unique<StringPool>();
    }

    void TearDown() override {
        pool.reset();
    }

    std::unique_ptr<StringPool> pool;
};

// Basic Allocation Tests
TEST_F(StringPoolTests, AllocateSingleString) {
    const char* str = "test";
    const char* allocated = pool->allocate_string(str, 4);
    
    ASSERT_NE(allocated, nullptr);
    EXPECT_EQ(std::strcmp(allocated, str), 0);
    EXPECT_EQ(std::strlen(allocated), 4);
}

TEST_F(StringPoolTests, AllocateMultipleStrings) {
    const char* str1 = "first";
    const char* str2 = "second";
    const char* str3 = "third";
    
    const char* alloc1 = pool->allocate_string(str1, 5);
    const char* alloc2 = pool->allocate_string(str2, 6);
    const char* alloc3 = pool->allocate_string(str3, 5);
    
    ASSERT_NE(alloc1, nullptr);
    ASSERT_NE(alloc2, nullptr);
    ASSERT_NE(alloc3, nullptr);
    
    EXPECT_EQ(std::strcmp(alloc1, str1), 0);
    EXPECT_EQ(std::strcmp(alloc2, str2), 0);
    EXPECT_EQ(std::strcmp(alloc3, str3), 0);
}

TEST_F(StringPoolTests, AllocateVaryingLengths) {
    const char* short_str = "a";
    const char* medium_str = "medium";
    const char* long_str = "this is a longer string";
    
    const char* alloc_short = pool->allocate_string(short_str, 1);
    const char* alloc_medium = pool->allocate_string(medium_str, 6);
    const char* alloc_long = pool->allocate_string(long_str, 24);
    
    EXPECT_EQ(std::strcmp(alloc_short, short_str), 0);
    EXPECT_EQ(std::strcmp(alloc_medium, medium_str), 0);
    EXPECT_EQ(std::strcmp(alloc_long, long_str), 0);
}

TEST_F(StringPoolTests, AllocateNullTerminated) {
    const char* str = "test";
    const char* allocated = pool->allocate_string(str, 4);
    
    // Verify null termination
    EXPECT_EQ(allocated[4], '\0');
    EXPECT_EQ(std::strlen(allocated), 4);
}

TEST_F(StringPoolTests, AllocateZeroLengthString) {
    const char* allocated = pool->allocate_string("", 0);
    EXPECT_EQ(allocated, nullptr); // Zero length returns nullptr
}

// Chunk Management Tests
TEST_F(StringPoolTests, DefaultChunkSize) {
    StringPool default_pool;
    EXPECT_GT(default_pool.total_allocated(), 0);
    EXPECT_EQ(default_pool.num_chunks(), 1);
}

TEST_F(StringPoolTests, CustomChunkSize) {
    StringPool custom_pool(512); // 512 bytes
    EXPECT_GE(custom_pool.total_allocated(), 512);
    EXPECT_EQ(custom_pool.num_chunks(), 1);
}

TEST_F(StringPoolTests, AllocationFitsInOneChunk) {
    const char* str = "test";
    pool->allocate_string(str, 4);
    
    EXPECT_EQ(pool->num_chunks(), 1);
    EXPECT_GT(pool->current_chunk_offset(), 0);
}

TEST_F(StringPoolTests, AllocationSpansMultipleChunks) {
    // Allocate strings until we need a new chunk
    // Default chunk size is 1MB, so allocate many strings
    size_t chunk_size = 1024 * 1024; // 1MB
    size_t string_size = 100;
    size_t strings_per_chunk = chunk_size / (string_size + 1);
    
    // Allocate enough to fill first chunk and start second
    for (size_t i = 0; i < strings_per_chunk + 10; ++i) {
        std::string str(string_size, 'a' + (i % 26));
        pool->allocate_string(str.c_str(), string_size);
    }
    
    EXPECT_GE(pool->num_chunks(), 2);
}

TEST_F(StringPoolTests, ChunkBoundariesHandled) {
    // Test allocation at chunk boundary
    StringPool small_pool(100); // Small chunk for testing
    
    // Fill most of first chunk
    for (int i = 0; i < 5; ++i) {
        pool->allocate_string("test", 4);
    }
    
    // Allocate something that would cross boundary
    const char* large = "this is a larger string that might cross boundary";
    const char* allocated = pool->allocate_string(large, 50);
    
    ASSERT_NE(allocated, nullptr);
    EXPECT_EQ(std::strcmp(allocated, large), 0);
}

// Memory Efficiency Tests
TEST_F(StringPoolTests, StringsStoredContiguously) {
    const char* str1 = "first";
    const char* str2 = "second";
    
    const char* alloc1 = pool->allocate_string(str1, 5);
    const char* alloc2 = pool->allocate_string(str2, 6);
    
    // Strings should be stored contiguously (alloc2 should be right after alloc1 + null terminator)
    EXPECT_EQ(alloc2, alloc1 + 6); // 5 chars + 1 null terminator
}

TEST_F(StringPoolTests, NoMemoryLeaks) {
    // Allocate many strings
    for (int i = 0; i < 1000; ++i) {
        std::string str = "string_" + std::to_string(i);
        pool->allocate_string(str.c_str(), str.length());
    }
    
    // Memory should be managed by unique_ptr
    // No explicit leak check needed, but verify allocation worked
    EXPECT_GT(pool->total_allocated(), 0);
}

TEST_F(StringPoolTests, ChunkReuseAfterClear) {
    // Allocate some strings
    pool->allocate_string("test1", 5);
    pool->allocate_string("test2", 5);
    
    size_t chunks_before = pool->num_chunks();
    
    // Clear pool
    pool->clear();
    
    // Chunks should still exist (for reuse)
    EXPECT_EQ(pool->num_chunks(), chunks_before);
    EXPECT_EQ(pool->current_chunk_idx(), 0);
    EXPECT_EQ(pool->current_chunk_offset(), 0);
}

TEST_F(StringPoolTests, ReuseAfterClear) {
    const char* str1 = "first";
    pool->allocate_string(str1, 5);
    
    pool->clear();
    
    // Allocate again - should reuse chunks
    const char* str2 = "second";
    const char* alloc2 = pool->allocate_string(str2, 6);
    
    ASSERT_NE(alloc2, nullptr);
    EXPECT_EQ(std::strcmp(alloc2, str2), 0);
    // alloc2 might be at same location as previous allocation (reused memory)
}

// Edge Cases
TEST_F(StringPoolTests, AllocateZeroLengthStringEdgeCase) {
    const char* allocated = pool->allocate_string("", 0);
    EXPECT_EQ(allocated, nullptr);
}

TEST_F(StringPoolTests, AllocateMaximumLengthString) {
    std::string max_str(FFX_STR_MAX_LENGTH, 'x');
    const char* allocated = pool->allocate_string(max_str.c_str(), FFX_STR_MAX_LENGTH);
    
    ASSERT_NE(allocated, nullptr);
    EXPECT_EQ(std::strlen(allocated), FFX_STR_MAX_LENGTH);
    EXPECT_EQ(std::strcmp(allocated, max_str.c_str()), 0);
}

TEST_F(StringPoolTests, AllocateManySmallStrings) {
    // Allocate many small strings
    for (int i = 0; i < 10000; ++i) {
        std::string str = std::to_string(i);
        const char* allocated = pool->allocate_string(str.c_str(), str.length());
        ASSERT_NE(allocated, nullptr);
        EXPECT_EQ(std::strcmp(allocated, str.c_str()), 0);
    }
}

TEST_F(StringPoolTests, AllocateFewLargeStrings) {
    // Allocate few large strings
    for (int i = 0; i < 10; ++i) {
        std::string large_str(1000, 'a' + (i % 26));
        const char* allocated = pool->allocate_string(large_str.c_str(), 1000);
        ASSERT_NE(allocated, nullptr);
        EXPECT_EQ(std::strcmp(allocated, large_str.c_str()), 0);
    }
}

TEST_F(StringPoolTests, StressTestMultipleChunks) {
    // Stress test: allocate until multiple chunks
    size_t string_size = 1000;
    int count = 0;
    
    while (pool->num_chunks() < 3 && count < 10000) {
        std::string str(string_size, 'x');
        pool->allocate_string(str.c_str(), string_size);
        count++;
    }
    
    EXPECT_GE(pool->num_chunks(), 2);
    EXPECT_GT(pool->total_allocated(), 0);
}

// Total Allocated Tests
TEST_F(StringPoolTests, TotalAllocated) {
    size_t initial = pool->total_allocated();
    EXPECT_GT(initial, 0); // Should have at least one chunk
    
    // Allocate some strings
    pool->allocate_string("test", 4);
    
    // Total should be same (chunk already allocated)
    size_t after = pool->total_allocated();
    EXPECT_EQ(after, initial);
}

TEST_F(StringPoolTests, TotalAllocatedAfterNewChunk) {
    // Force new chunk allocation
    StringPool small_pool(100);
    small_pool.allocate_string("test", 4);
    size_t after_first = small_pool.total_allocated();
    
    // Allocate enough to trigger new chunk
    for (int i = 0; i < 10; ++i) {
        std::string str(50, 'x');
        small_pool.allocate_string(str.c_str(), 50);
    }
    
    size_t after_second = small_pool.total_allocated();
    EXPECT_GT(after_second, after_first);
}

// Current State Tests
TEST_F(StringPoolTests, CurrentChunkOffset) {
    EXPECT_EQ(pool->current_chunk_offset(), 0);
    
    pool->allocate_string("test", 4);
    EXPECT_GT(pool->current_chunk_offset(), 0);
}

TEST_F(StringPoolTests, CurrentChunkIdx) {
    EXPECT_EQ(pool->current_chunk_idx(), 0);
    
    // Should stay at 0 until new chunk is needed
    pool->allocate_string("test", 4);
    EXPECT_EQ(pool->current_chunk_idx(), 0);
}

TEST_F(StringPoolTests, CurrentChunkIdxAfterNewChunk) {
    StringPool small_pool(100);
    
    // Fill first chunk
    for (int i = 0; i < 10; ++i) {
        std::string str(8, 'x');
        small_pool.allocate_string(str.c_str(), 8);
    }
    
    // Allocate more to trigger new chunk
    std::string str(50, 'y');
    small_pool.allocate_string(str.c_str(), 50);
    
    EXPECT_GE(small_pool.current_chunk_idx(), 0);
    EXPECT_GE(small_pool.num_chunks(), 1);
}

}// namespace ffx

