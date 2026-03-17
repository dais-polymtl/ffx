#include "../src/table/include/string_dictionary.hpp"
#include "../src/table/include/string_pool.hpp"
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace ffx {

TEST(StringDictionaryMTTests, ParallelLookups) {
    StringPool pool;
    StringDictionary dict(&pool);

    const int num_unique_strings = 1000;
    std::vector<std::string> strings;
    for (int i = 0; i < num_unique_strings; ++i) {
        strings.push_back("multi_threaded_test_string_" + std::to_string(i));
        dict.add_string(ffx_str_t(strings[i], &pool));
    }
    dict.finalize();

    // Pre-create lookup strings safely before parallel lookups
    std::vector<ffx_str_t> lookup_ffx_strs;
    for (int i = 0; i < num_unique_strings; ++i) {
        lookup_ffx_strs.push_back(ffx_str_t(strings[i], &pool));
    }

    const int num_threads = 12;
    const int lookups_per_thread = 50000;
    std::vector<std::thread> workers;
    std::atomic<bool> start{false};
    std::atomic<int> ready_count{0};

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            ready_count++;
            while (!start) {
                std::this_thread::yield();
            }

            for (int i = 0; i < lookups_per_thread; ++i) {
                int idx = (i + t) % num_unique_strings;

                // 1. Test get_id (hash table lookup - READ ONLY)
                uint64_t id = dict.get_id(lookup_ffx_strs[idx]);
                ASSERT_EQ(id, static_cast<uint64_t>(idx));

                // 2. Test get_string (vector lookup - READ ONLY)
                const ffx_str_t& retrieved = dict.get_string(id);
                EXPECT_EQ(retrieved.to_string(), strings[idx]);
            }
        });
    }

    // Wait for all threads to be ready
    while (ready_count < num_threads) {
        std::this_thread::yield();
    }

    // Start the race!
    start = true;

    for (auto& w: workers) {
        w.join();
    }
}

TEST(StringDictionaryMTTests, ParallelLookupsWithEmptyAndNull) {
    StringPool pool;
    StringDictionary dict(&pool);

    dict.add_string(ffx_str_t::null_value());
    dict.add_string(ffx_str_t("", &pool));
    dict.add_string(ffx_str_t("some_string", &pool));
    dict.finalize();

    ffx_str_t null_str = ffx_str_t::null_value();
    ffx_str_t empty_str = ffx_str_t("", &pool);
    ffx_str_t normal_str = ffx_str_t("some_string", &pool);

    const int num_threads = 8;
    const int lookups_per_thread = 50000;
    std::vector<std::thread> workers;

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            for (int i = 0; i < lookups_per_thread; ++i) {
                // Null lookup
                EXPECT_TRUE(dict.get_string(0).is_null());
                EXPECT_EQ(dict.get_id(null_str), 0u);

                // Empty lookup
                EXPECT_TRUE(dict.get_string(1).is_empty_string());
                EXPECT_EQ(dict.get_id(empty_str), 1u);

                // Normal lookup
                EXPECT_EQ(dict.get_string(2).to_string(), "some_string");
                EXPECT_EQ(dict.get_id(normal_str), 2u);
            }
        });
    }

    for (auto& w: workers) {
        w.join();
    }
}

}// namespace ffx
