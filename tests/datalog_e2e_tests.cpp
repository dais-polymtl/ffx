#include <gtest/gtest.h>
#include "../src/table/include/table_loader.hpp"
#include "../src/table/include/table_loader_utils.hpp"
#include "../src/operator/include/schema/adj_list_manager.hpp"
#include "../src/operator/include/schema/schema.hpp"
#include "../src/query/include/query.hpp"
#include "../src/ser_der/include/serializer.hpp"
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

class DatalogE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directories for two tables
        table1_dir = fs::temp_directory_path() / "datalog_e2e_table1";
        table2_dir = fs::temp_directory_path() / "datalog_e2e_table2";
        fs::create_directories(table1_dir);
        fs::create_directories(table2_dir);
        
        setup_table1();
        setup_table2();
    }

    void TearDown() override {
        if (fs::exists(table1_dir)) {
            fs::remove_all(table1_dir);
        }
        if (fs::exists(table2_dir)) {
            fs::remove_all(table2_dir);
        }
    }

    void setup_table1() {
        // Table T1: Knows(person_id, friend_id, timestamp, location)
        // person_id: uint64, friend_id: uint64, timestamp: uint64, location: string
        // Data:
        //   0 -> 1, timestamp=100, location="NYC"
        //   0 -> 2, timestamp=200, location="SF"
        //   1 -> 2, timestamp=300, location="LA"
        
        std::vector<uint64_t> person_id = {0, 0, 1};
        std::vector<uint64_t> friend_id = {1, 2, 2};
        std::vector<uint64_t> timestamp = {100, 200, 300};
        std::vector<std::string> location = {"NYC", "SF", "LA"};
        std::vector<bool> location_null = {false, false, false};
        
        // Serialize columns
        uint64_t max_person = 1, max_friend = 2, max_timestamp = 300;
        ffx::serialize_uint64_column(person_id, table1_dir + "/person_id_uint64.bin", &max_person);
        ffx::serialize_uint64_column(friend_id, table1_dir + "/friend_id_uint64.bin", &max_friend);
        ffx::serialize_uint64_column(timestamp, table1_dir + "/timestamp_uint64.bin", &max_timestamp);
        ffx::serialize_string_column(location, location_null, table1_dir + "/location_string.bin");
        
        // Create metadata
        std::vector<ffx::ColumnConfig> configs;
        configs.emplace_back("person_id", 0, ffx::ColumnType::UINT64);
        configs.emplace_back("friend_id", 1, ffx::ColumnType::UINT64);
        configs.emplace_back("timestamp", 2, ffx::ColumnType::UINT64);
        configs.emplace_back("location", 3, ffx::ColumnType::STRING);
        std::vector<uint64_t> max_values = {max_person, max_friend, max_timestamp, 0};
        ffx::write_metadata_binary(table1_dir, configs, 3, max_values);
    }

    void setup_table2() {
        // Table T2: LivesIn(city_id, person_id, country, population)
        // city_id: string, person_id: uint64, country: string, population: uint64
        // Data:
        //   "NYC" -> 0, country="USA", population=8000000
        //   "SF" -> 1, country="USA", population=800000
        //   "LA" -> 2, country="USA", population=4000000
        
        std::vector<std::string> city_id = {"NYC", "SF", "LA"};
        std::vector<bool> city_id_null = {false, false, false};
        std::vector<uint64_t> person_id = {0, 1, 2};
        std::vector<std::string> country = {"USA", "USA", "USA"};
        std::vector<bool> country_null = {false, false, false};
        std::vector<uint64_t> population = {8000000, 800000, 4000000};
        
        // Serialize columns
        uint64_t max_person = 2, max_population = 8000000;
        ffx::serialize_string_column(city_id, city_id_null, table2_dir + "/city_id_string.bin");
        ffx::serialize_uint64_column(person_id, table2_dir + "/person_id_uint64.bin", &max_person);
        ffx::serialize_string_column(country, country_null, table2_dir + "/country_string.bin");
        ffx::serialize_uint64_column(population, table2_dir + "/population_uint64.bin", &max_population);
        
        // Create metadata
        std::vector<ffx::ColumnConfig> configs;
        configs.emplace_back("city_id", 0, ffx::ColumnType::STRING);
        configs.emplace_back("person_id", 1, ffx::ColumnType::UINT64);
        configs.emplace_back("country", 2, ffx::ColumnType::STRING);
        configs.emplace_back("population", 3, ffx::ColumnType::UINT64);
        std::vector<uint64_t> max_values = {0, max_person, 0, max_population};
        ffx::write_metadata_binary(table2_dir, configs, 3, max_values);
    }

    std::string table1_dir;
    std::string table2_dir;
};

TEST_F(DatalogE2ETest, MultiTableJoinWithUnderscoreAndString) {
    // Query: T1(p1, f1, _, _), T2(c1, f1, _, _)
    // This means:
    //   T1: person_id=p1, friend_id=f1, ignore timestamp, ignore location
    //   T2: city_id=c1, person_id=f1, ignore country, ignore population
    // Join on: T1.friend_id = T2.person_id (f1)
    // Last join attribute (T2.city_id) is a string
    
    ffx::Query query{"Q(p1, f1, c1) := T1(p1, f1, _, _), T2(c1, f1, _, _)"};
    ASSERT_TRUE(query.is_datalog_format());
    ASSERT_EQ(query.num_rels, 2);
    
    ffx::AdjListManager adj_list_manager;
    
    // Load table T1
    auto metadata1 = ffx::read_metadata_binary(table1_dir);
    std::vector<std::string> table1_columns;
    for (const auto& col : metadata1.columns) {
        table1_columns.push_back(col.attr_name);
    }
    
    // Map T1: p1 (pos 0) -> person_id, f1 (pos 1) -> friend_id
    auto [t1_src, t1_dest] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        table1_columns,
        "T1"
    );
    EXPECT_EQ(t1_src, "person_id");
    EXPECT_EQ(t1_dest, "friend_id");
    
    // Load T1
    ffx::LoadedTable table1 = ffx::load_table_from_columns_auto(
        table1_dir,
        "T1",
        table1_columns,
        query,
        adj_list_manager
    );
    
    EXPECT_EQ(table1.num_rows, 3);
    EXPECT_NE(table1.get_uint64_column("person_id"), nullptr);
    EXPECT_NE(table1.get_uint64_column("friend_id"), nullptr);
    
    // Load table T2
    auto metadata2 = ffx::read_metadata_binary(table2_dir);
    std::vector<std::string> table2_columns;
    for (const auto& col : metadata2.columns) {
        table2_columns.push_back(col.attr_name);
    }
    
    // Map T2: c1 (pos 0) -> city_id (STRING!), f1 (pos 1) -> person_id
    auto [t2_src, t2_dest] = ffx::map_datalog_to_table_columns(
        query.rels[1],
        table2_columns,
        "T2"
    );
    EXPECT_EQ(t2_src, "city_id");  // String column!
    EXPECT_EQ(t2_dest, "person_id");
    
    // Load T2 (with string join attribute)
    ffx::LoadedTable table2 = ffx::load_table_from_columns_auto(
        table2_dir,
        "T2",
        table2_columns,
        query,
        adj_list_manager
    );
    
    EXPECT_EQ(table2.num_rows, 3);
    EXPECT_NE(table2.get_string_column("city_id"), nullptr);
    EXPECT_NE(table2.get_uint64_column("person_id"), nullptr);
    
    // Verify string dictionary was built
    EXPECT_NE(table2.global_string_dict.get(), nullptr);
    EXPECT_GT(table2.global_string_dict->size(), 0);
    
    // Verify adjacency lists were built for both tables
    EXPECT_TRUE(adj_list_manager.is_loaded(table1_dir, "person_id", "friend_id", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(table2_dir, "city_id", "person_id", true));
    
    // Verify T2 has string-based adjacency lists
    auto* t2_fwd = adj_list_manager.get_or_load(table2_dir, "city_id", "person_id", true);
    EXPECT_NE(t2_fwd, nullptr);
    
    // Populate schema for both tables
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    // Populate T1
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table1_dir, "person_id", "friend_id", adj_list_manager
    );
    
    // Populate T2 (string join)
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table2_dir, "city_id", "person_id", adj_list_manager
    );
    
    // Verify schema has both adjacency lists
    EXPECT_TRUE(schema.has_adj_list("person_id", "friend_id"));
    EXPECT_TRUE(schema.has_adj_list("friend_id", "person_id"));
    EXPECT_TRUE(schema.has_adj_list("city_id", "person_id"));
    EXPECT_TRUE(schema.has_adj_list("person_id", "city_id"));
}

TEST_F(DatalogE2ETest, ThreeTableJoinWithMultipleColumns) {
    // Create a third table T3
    std::string table3_dir = fs::temp_directory_path() / "datalog_e2e_table3";
    fs::create_directories(table3_dir);
    
    // Table T3: WorksAt(company, person_id, salary, department)
    // company: string, person_id: uint64, salary: uint64, department: string
    std::vector<std::string> company = {"Google", "Apple", "Microsoft"};
    std::vector<bool> company_null = {false, false, false};
    std::vector<uint64_t> person_id = {0, 1, 2};
    std::vector<uint64_t> salary = {100000, 120000, 110000};
    std::vector<std::string> department = {"Eng", "Eng", "Sales"};
    std::vector<bool> dept_null = {false, false, false};
    
    uint64_t max_person = 2, max_salary = 120000;
    ffx::serialize_string_column(company, company_null, table3_dir + "/company_string.bin");
    ffx::serialize_uint64_column(person_id, table3_dir + "/person_id_uint64.bin", &max_person);
    ffx::serialize_uint64_column(salary, table3_dir + "/salary_uint64.bin", &max_salary);
    ffx::serialize_string_column(department, dept_null, table3_dir + "/department_string.bin");
    
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("company", 0, ffx::ColumnType::STRING);
    configs.emplace_back("person_id", 1, ffx::ColumnType::UINT64);
    configs.emplace_back("salary", 2, ffx::ColumnType::UINT64);
    configs.emplace_back("department", 3, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values = {0, max_person, max_salary, 0};
    ffx::write_metadata_binary(table3_dir, configs, 3, max_values);
    
    // Query: T1(p1, f1), T2(c1, f1), T3(comp, f1)
    // Note: Query parser doesn't allow underscores in edge positions (0,1)
    // But we're testing that tables have >3 columns and we ignore the extra ones
    // Join chain: T1.friend_id = T2.person_id = T3.person_id
    // Last join: T3.company (string) -> T3.person_id
    
    ffx::Query query{"Q(p1, f1, c1, comp) := T1(p1, f1), T2(c1, f1), T3(comp, f1)"};
    ASSERT_EQ(query.num_rels, 3);
    
    ffx::AdjListManager adj_list_manager;
    
    // Load all three tables
    auto metadata1 = ffx::read_metadata_binary(table1_dir);
    std::vector<std::string> t1_cols;
    for (const auto& col : metadata1.columns) t1_cols.push_back(col.attr_name);
    
    auto metadata2 = ffx::read_metadata_binary(table2_dir);
    std::vector<std::string> t2_cols;
    for (const auto& col : metadata2.columns) t2_cols.push_back(col.attr_name);
    
    auto metadata3 = ffx::read_metadata_binary(table3_dir);
    std::vector<std::string> t3_cols;
    for (const auto& col : metadata3.columns) t3_cols.push_back(col.attr_name);
    
    // Load T1
    ffx::LoadedTable t1 = ffx::load_table_from_columns_auto(
        table1_dir, "T1", t1_cols, query, adj_list_manager
    );
    
    // Load T2 (with underscore in first position)
    ffx::LoadedTable t2 = ffx::load_table_from_columns_auto(
        table2_dir, "T2", t2_cols, query, adj_list_manager
    );
    
    // Load T3 (string join attribute)
    ffx::LoadedTable t3 = ffx::load_table_from_columns_auto(
        table3_dir, "T3", t3_cols, query, adj_list_manager
    );
    
    // Verify all loaded
    EXPECT_EQ(t1.num_rows, 3);
    EXPECT_EQ(t2.num_rows, 3);
    EXPECT_EQ(t3.num_rows, 3);
    
    // Verify T3 has string dictionary
    EXPECT_NE(t3.global_string_dict.get(), nullptr);
    
    // Verify adjacency lists for all three
    EXPECT_TRUE(adj_list_manager.is_loaded(table1_dir, "person_id", "friend_id", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(table2_dir, "city_id", "person_id", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(table3_dir, "company", "person_id", true));
    
    // Populate schema
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table1_dir, "person_id", "friend_id", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table2_dir, "city_id", "person_id", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table3_dir, "company", "person_id", adj_list_manager
    );
    
    // Verify all adjacency lists in schema
    EXPECT_TRUE(schema.has_adj_list("person_id", "friend_id"));
    EXPECT_TRUE(schema.has_adj_list("city_id", "person_id"));
    EXPECT_TRUE(schema.has_adj_list("company", "person_id"));
    
    // Cleanup
    if (fs::exists(table3_dir)) {
        fs::remove_all(table3_dir);
    }
}

TEST_F(DatalogE2ETest, StringJoinWithMultipleColumns) {
    // Query: T1(p1, f1), T2(c1, f1)
    // Note: Query parser requires actual variables in edge positions (0,1)
    // But we're testing tables with >3 columns where we only use first 2 for joins
    // T1: person_id, friend_id (has 4 columns total: person_id, friend_id, timestamp, location)
    // T2: city_id (string), person_id (has 4 columns total: city_id, person_id, country, population)
    // Join on: T1.friend_id = T2.person_id
    // Last join attribute (T2.city_id) is a string
    
    ffx::Query query{"Q(p1, f1, c1) := T1(p1, f1), T2(c1, f1)"};
    
    ffx::AdjListManager adj_list_manager;
    
    // Load T1 - map by position
    // Position 0 (p1) -> person_id, Position 1 (f1) -> friend_id
    // Note: T1 has 4 columns total, but we only use first 2 for the join
    auto metadata1 = ffx::read_metadata_binary(table1_dir);
    std::vector<std::string> t1_cols;
    for (const auto& col : metadata1.columns) t1_cols.push_back(col.attr_name);
    
    // Verify T1 has 4 columns
    EXPECT_EQ(t1_cols.size(), 4);
    EXPECT_EQ(t1_cols[0], "person_id");
    EXPECT_EQ(t1_cols[1], "friend_id");
    EXPECT_EQ(t1_cols[2], "timestamp");
    EXPECT_EQ(t1_cols[3], "location");
    
    auto [t1_src, t1_dest] = ffx::map_datalog_to_table_columns(
        query.rels[0],
        t1_cols,
        "T1"
    );
    EXPECT_EQ(t1_src, "person_id");
    EXPECT_EQ(t1_dest, "friend_id");
    
    // Load T2 with string join
    // Note: T2 has 4 columns total, but we only use first 2 for the join
    auto metadata2 = ffx::read_metadata_binary(table2_dir);
    std::vector<std::string> t2_cols;
    for (const auto& col : metadata2.columns) t2_cols.push_back(col.attr_name);
    
    // Verify T2 has 4 columns
    EXPECT_EQ(t2_cols.size(), 4);
    EXPECT_EQ(t2_cols[0], "city_id");  // String - this is the join attribute!
    EXPECT_EQ(t2_cols[1], "person_id");
    EXPECT_EQ(t2_cols[2], "country");
    EXPECT_EQ(t2_cols[3], "population");
    
    auto [t2_src, t2_dest] = ffx::map_datalog_to_table_columns(
        query.rels[1],
        t2_cols,
        "T2"
    );
    EXPECT_EQ(t2_src, "city_id");  // String!
    EXPECT_EQ(t2_dest, "person_id");
    
    // Load both tables
    ffx::LoadedTable t1 = ffx::load_table_from_columns(
        table1_dir, t1_src, t1_dest, query, adj_list_manager
    );
    
    ffx::LoadedTable t2 = ffx::load_table_from_columns(
        table2_dir, t2_src, t2_dest, query, adj_list_manager
    );
    
    // Verify string dictionary for T2
    EXPECT_NE(t2.global_string_dict.get(), nullptr);
    EXPECT_GT(t2.global_string_dict->size(), 0);
    
    // Verify string ID column exists
    const uint64_t* city_ids = t2.get_string_id_column("city_id");
    ASSERT_NE(city_ids, nullptr);
    
    // Verify adjacency lists
    EXPECT_TRUE(adj_list_manager.is_loaded(table1_dir, "person_id", "friend_id", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(table2_dir, "city_id", "person_id", true));
    
    // Verify that extra columns (beyond first 2) are present in metadata but not loaded
    // This simulates what underscores would do - ignore extra columns
    EXPECT_EQ(t1_cols.size(), 4);  // person_id, friend_id, timestamp, location
    EXPECT_EQ(t2_cols.size(), 4);  // city_id, person_id, country, population
    
    // Only the join columns should be loaded
    EXPECT_NE(t1.get_uint64_column("person_id"), nullptr);
    EXPECT_NE(t1.get_uint64_column("friend_id"), nullptr);
    // Extra columns (timestamp, location) are not loaded - this is what underscores represent
    
    EXPECT_NE(t2.get_string_column("city_id"), nullptr);
    EXPECT_NE(t2.get_uint64_column("person_id"), nullptr);
    // Extra columns (country, population) are not loaded
}

TEST_F(DatalogE2ETest, FourTableJoinWithStringJoinAndMultipleColumns) {
    // Most comprehensive test: 4 tables, all with >3 columns, string join in last table
    // Create T3 and T4
    std::string table3_dir = fs::temp_directory_path() / "datalog_e2e_table3";
    std::string table4_dir = fs::temp_directory_path() / "datalog_e2e_table4";
    fs::create_directories(table3_dir);
    fs::create_directories(table4_dir);
    
    // Setup T3: WorksAt(company, person_id, salary, department)
    std::vector<std::string> company = {"Google", "Apple", "Microsoft"};
    std::vector<bool> company_null = {false, false, false};
    std::vector<uint64_t> person_id = {0, 1, 2};
    std::vector<uint64_t> salary = {100000, 120000, 110000};
    std::vector<std::string> department = {"Eng", "Eng", "Sales"};
    std::vector<bool> dept_null = {false, false, false};
    
    uint64_t max_person = 2, max_salary = 120000;
    ffx::serialize_string_column(company, company_null, table3_dir + "/company_string.bin");
    ffx::serialize_uint64_column(person_id, table3_dir + "/person_id_uint64.bin", &max_person);
    ffx::serialize_uint64_column(salary, table3_dir + "/salary_uint64.bin", &max_salary);
    ffx::serialize_string_column(department, dept_null, table3_dir + "/department_string.bin");
    
    std::vector<ffx::ColumnConfig> configs3;
    configs3.emplace_back("company", 0, ffx::ColumnType::STRING);
    configs3.emplace_back("person_id", 1, ffx::ColumnType::UINT64);
    configs3.emplace_back("salary", 2, ffx::ColumnType::UINT64);
    configs3.emplace_back("department", 3, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values3 = {0, max_person, max_salary, 0};
    ffx::write_metadata_binary(table3_dir, configs3, 3, max_values3);
    
    // Table T4: LocatedIn(city_name, country_name, region, timezone)
    // city_name: string, country_name: string, region: string, timezone: uint64
    // This table has a string-to-string join (city_name -> country_name)
    std::vector<std::string> city_name = {"NYC", "SF", "LA"};
    std::vector<bool> city_name_null = {false, false, false};
    std::vector<std::string> country_name = {"USA", "USA", "USA"};
    std::vector<bool> country_name_null = {false, false, false};
    std::vector<std::string> region = {"East", "West", "West"};
    std::vector<bool> region_null = {false, false, false};
    std::vector<uint64_t> timezone = {5, 8, 8};
    
    uint64_t max_timezone = 8;
    ffx::serialize_string_column(city_name, city_name_null, table4_dir + "/city_name_string.bin");
    ffx::serialize_string_column(country_name, country_name_null, table4_dir + "/country_name_string.bin");
    ffx::serialize_string_column(region, region_null, table4_dir + "/region_string.bin");
    ffx::serialize_uint64_column(timezone, table4_dir + "/timezone_uint64.bin", &max_timezone);
    
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("city_name", 0, ffx::ColumnType::STRING);
    configs.emplace_back("country_name", 1, ffx::ColumnType::STRING);
    configs.emplace_back("region", 2, ffx::ColumnType::STRING);
    configs.emplace_back("timezone", 3, ffx::ColumnType::UINT64);
    std::vector<uint64_t> max_values = {0, 0, 0, max_timezone};
    ffx::write_metadata_binary(table4_dir, configs, 3, max_values);
    
    // Query: T1(p1, f1), T2(c1, f1), T3(comp, f1), T4(city, country)
    // Join chain: T1.friend_id = T2.person_id = T3.person_id, T2.city_id = T4.city_name
    // Last join: T4.city_name (string) -> T4.country_name (string) - both are strings!
    
    ffx::Query query{"Q(p1, f1, c1, comp, city, country) := "
                     "T1(p1, f1), T2(c1, f1), T3(comp, f1), T4(city, country)"};
    ASSERT_EQ(query.num_rels, 4);
    
    ffx::AdjListManager adj_list_manager;
    
    // Load all four tables
    auto metadata1 = ffx::read_metadata_binary(table1_dir);
    std::vector<std::string> t1_cols;
    for (const auto& col : metadata1.columns) t1_cols.push_back(col.attr_name);
    
    auto metadata2 = ffx::read_metadata_binary(table2_dir);
    std::vector<std::string> t2_cols;
    for (const auto& col : metadata2.columns) t2_cols.push_back(col.attr_name);
    
    auto metadata3 = ffx::read_metadata_binary(table3_dir);
    std::vector<std::string> t3_cols;
    for (const auto& col : metadata3.columns) t3_cols.push_back(col.attr_name);
    
    auto metadata4 = ffx::read_metadata_binary(table4_dir);
    std::vector<std::string> t4_cols;
    for (const auto& col : metadata4.columns) t4_cols.push_back(col.attr_name);
    
    // Verify all tables have >3 columns
    EXPECT_GT(t1_cols.size(), 3);
    EXPECT_GT(t2_cols.size(), 3);
    EXPECT_GT(t3_cols.size(), 3);
    EXPECT_GT(t4_cols.size(), 3);
    
    // Load all tables
    ffx::LoadedTable t1 = ffx::load_table_from_columns_auto(
        table1_dir, "T1", t1_cols, query, adj_list_manager
    );
    ffx::LoadedTable t2 = ffx::load_table_from_columns_auto(
        table2_dir, "T2", t2_cols, query, adj_list_manager
    );
    ffx::LoadedTable t3 = ffx::load_table_from_columns_auto(
        table3_dir, "T3", t3_cols, query, adj_list_manager
    );
    ffx::LoadedTable t4 = ffx::load_table_from_columns_auto(
        table4_dir, "T4", t4_cols, query, adj_list_manager
    );
    
    // Verify T4 has string-to-string join (both src and dest are strings)
    auto [t4_src, t4_dest] = ffx::map_datalog_to_table_columns(
        query.rels[3],
        t4_cols,
        "T4"
    );
    EXPECT_EQ(t4_src, "city_name");  // String
    EXPECT_EQ(t4_dest, "country_name");  // String
    
    // Verify string dictionaries
    EXPECT_NE(t2.global_string_dict.get(), nullptr);
    EXPECT_NE(t3.global_string_dict.get(), nullptr);
    EXPECT_NE(t4.global_string_dict.get(), nullptr);
    
    // Verify adjacency lists for all tables
    EXPECT_TRUE(adj_list_manager.is_loaded(table1_dir, "person_id", "friend_id", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(table2_dir, "city_id", "person_id", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(table3_dir, "company", "person_id", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(table4_dir, "city_name", "country_name", true));
    
    // Populate schema
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table1_dir, "person_id", "friend_id", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table2_dir, "city_id", "person_id", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table3_dir, "company", "person_id", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table4_dir, "city_name", "country_name", adj_list_manager
    );
    
    // Verify all adjacency lists in schema
    EXPECT_TRUE(schema.has_adj_list("person_id", "friend_id"));
    EXPECT_TRUE(schema.has_adj_list("city_id", "person_id"));
    EXPECT_TRUE(schema.has_adj_list("company", "person_id"));
    EXPECT_TRUE(schema.has_adj_list("city_name", "country_name"));  // String-to-string!
    EXPECT_TRUE(schema.has_adj_list("country_name", "city_name"));
    
    // Cleanup
    if (fs::exists(table3_dir)) {
        fs::remove_all(table3_dir);
    }
    if (fs::exists(table4_dir)) {
        fs::remove_all(table4_dir);
    }
}

TEST_F(DatalogE2ETest, StringJoinWithNulls) {
    // Test string-to-string join with NULL values
    // Table T1: Person(name, friend_name)
    // name: string, friend_name: string (with NULLs)
    // Data:
    //   "Alice" -> "Bob"
    //   "Bob" -> NULL
    //   "Charlie" -> "Alice"
    
    std::string table3_dir = fs::temp_directory_path() / "datalog_e2e_table3_nulls";
    fs::create_directories(table3_dir);
    
    std::vector<std::string> name = {"Alice", "Bob", "Charlie"};
    std::vector<bool> name_null = {false, false, false};
    std::vector<std::string> friend_name = {"Bob", "", "Alice"};  // Empty = NULL
    std::vector<bool> friend_name_null = {false, true, false};
    
    ffx::serialize_string_column(name, name_null, table3_dir + "/name_string.bin");
    ffx::serialize_string_column(friend_name, friend_name_null, table3_dir + "/friend_name_string.bin");
    
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("name", 0, ffx::ColumnType::STRING);
    configs.emplace_back("friend_name", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values = {0, 0};
    ffx::write_metadata_binary(table3_dir, configs, 3, max_values);
    
    // Query: T3(n, f)
    ffx::Query query{"Q(n, f) := T3(n, f)"};
    
    ffx::AdjListManager adj_list_manager;
    
    auto metadata = ffx::read_metadata_binary(table3_dir);
    std::vector<std::string> t3_cols;
    for (const auto& col : metadata.columns) t3_cols.push_back(col.attr_name);
    
    // Load table
    ffx::LoadedTable t3 = ffx::load_table_from_columns_auto(
        table3_dir, "T3", t3_cols, query, adj_list_manager
    );
    
    EXPECT_EQ(t3.num_rows, 3);
    EXPECT_NE(t3.get_string_column("name"), nullptr);
    EXPECT_NE(t3.get_string_column("friend_name"), nullptr);
    
    // Verify string dictionary
    EXPECT_NE(t3.global_string_dict.get(), nullptr);
    EXPECT_GT(t3.global_string_dict->size(), 0);
    
    // Verify that NULL values are handled correctly
    const uint64_t* friend_name_ids = t3.get_string_id_column("friend_name");
    ASSERT_NE(friend_name_ids, nullptr);
    
    // Check that NULL (UINT64_MAX) is present in ID column
    // Bob's friend_name should be NULL (UINT64_MAX)
    bool found_null = false;
    for (uint64_t i = 0; i < t3.num_rows; i++) {
        if (friend_name_ids[i] == UINT64_MAX) {
            found_null = true;
            break;
        }
    }
    EXPECT_TRUE(found_null) << "Should have at least one NULL value";
    
    // Verify adjacency lists - NULL edges should be skipped
    EXPECT_TRUE(adj_list_manager.is_loaded(table3_dir, "name", "friend_name", true));
    
    auto* fwd_adj = adj_list_manager.get_or_load(table3_dir, "name", "friend_name", true);
    ASSERT_NE(fwd_adj, nullptr);
    
    // Verify that adjacency lists don't include NULL edges
    // Alice -> Bob should exist, Bob -> NULL should be skipped, Charlie -> Alice should exist
    // So we should have 2 edges total (not 3)
    
    // Populate schema
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table3_dir, "name", "friend_name", adj_list_manager
    );
    
    EXPECT_TRUE(schema.has_adj_list("name", "friend_name"));
    EXPECT_TRUE(schema.has_adj_list("friend_name", "name"));
    
    // Cleanup
    if (fs::exists(table3_dir)) {
        fs::remove_all(table3_dir);
    }
}

TEST_F(DatalogE2ETest, IntToStringJoinWithNulls) {
    // Test int-to-string join with NULL string values
    // Table T1: Employee(id, department)
    // id: uint64, department: string (with NULLs)
    // Data:
    //   0 -> "Engineering"
    //   1 -> NULL
    //   2 -> "Sales"
    
    std::string table4_dir = fs::temp_directory_path() / "datalog_e2e_table4_nulls";
    fs::create_directories(table4_dir);
    
    std::vector<uint64_t> id = {0, 1, 2};
    std::vector<std::string> department = {"Engineering", "", "Sales"};
    std::vector<bool> department_null = {false, true, false};
    
    uint64_t max_id = 2;
    ffx::serialize_uint64_column(id, table4_dir + "/id_uint64.bin", &max_id);
    ffx::serialize_string_column(department, department_null, table4_dir + "/department_string.bin");
    
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("id", 0, ffx::ColumnType::UINT64);
    configs.emplace_back("department", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values = {max_id, 0};
    ffx::write_metadata_binary(table4_dir, configs, 3, max_values);
    
    // Query: T4(i, d)
    ffx::Query query{"Q(i, d) := T4(i, d)"};
    
    ffx::AdjListManager adj_list_manager;
    
    auto metadata = ffx::read_metadata_binary(table4_dir);
    std::vector<std::string> t4_cols;
    for (const auto& col : metadata.columns) t4_cols.push_back(col.attr_name);
    
    // Load table
    ffx::LoadedTable t4 = ffx::load_table_from_columns_auto(
        table4_dir, "T4", t4_cols, query, adj_list_manager
    );
    
    EXPECT_EQ(t4.num_rows, 3);
    EXPECT_NE(t4.get_uint64_column("id"), nullptr);
    EXPECT_NE(t4.get_string_column("department"), nullptr);
    
    // Verify NULL handling
    const uint64_t* dept_ids = t4.get_string_id_column("department");
    ASSERT_NE(dept_ids, nullptr);
    
    // Check that NULL (UINT64_MAX) is present
    EXPECT_EQ(dept_ids[1], UINT64_MAX) << "Employee 1 should have NULL department";
    
    // Verify adjacency lists - NULL edges should be skipped
    EXPECT_TRUE(adj_list_manager.is_loaded(table4_dir, "id", "department", true));
    
    // Populate schema
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table4_dir, "id", "department", adj_list_manager
    );
    
    EXPECT_TRUE(schema.has_adj_list("id", "department"));
    EXPECT_TRUE(schema.has_adj_list("department", "id"));
    
    // Cleanup
    if (fs::exists(table4_dir)) {
        fs::remove_all(table4_dir);
    }
}

TEST_F(DatalogE2ETest, MultiTableJoinWithStringNulls) {
    // Test multi-table join with string columns containing NULLs
    // T1: Person(id, name) - id: uint64, name: string
    // T2: WorksAt(name, company) - name: string, company: string (with NULLs)
    // Join on: T1.name = T2.name
    
    std::string t1_dir = fs::temp_directory_path() / "datalog_e2e_t1_nulls";
    std::string t2_dir = fs::temp_directory_path() / "datalog_e2e_t2_nulls";
    fs::create_directories(t1_dir);
    fs::create_directories(t2_dir);
    
    // T1: Person(id, name)
    std::vector<uint64_t> id = {0, 1, 2};
    std::vector<std::string> name = {"Alice", "Bob", "Charlie"};
    std::vector<bool> name_null = {false, false, false};
    
    uint64_t max_id = 2;
    ffx::serialize_uint64_column(id, t1_dir + "/id_uint64.bin", &max_id);
    ffx::serialize_string_column(name, name_null, t1_dir + "/name_string.bin");
    
    std::vector<ffx::ColumnConfig> configs1;
    configs1.emplace_back("id", 0, ffx::ColumnType::UINT64);
    configs1.emplace_back("name", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values1 = {max_id, 0};
    ffx::write_metadata_binary(t1_dir, configs1, 3, max_values1);
    
    // T2: WorksAt(name, company) - company has NULLs
    std::vector<std::string> name2 = {"Alice", "Bob", "Charlie"};
    std::vector<bool> name2_null = {false, false, false};
    std::vector<std::string> company = {"Google", "", "Microsoft"};  // Bob has NULL company
    std::vector<bool> company_null = {false, true, false};
    
    ffx::serialize_string_column(name2, name2_null, t2_dir + "/name_string.bin");
    ffx::serialize_string_column(company, company_null, t2_dir + "/company_string.bin");
    
    std::vector<ffx::ColumnConfig> configs2;
    configs2.emplace_back("name", 0, ffx::ColumnType::STRING);
    configs2.emplace_back("company", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values2 = {0, 0};
    ffx::write_metadata_binary(t2_dir, configs2, 3, max_values2);
    
    // Query: T1(i, n), T2(n, c)
    ffx::Query query{"Q(i, n, c) := T1(i, n), T2(n, c)"};
    
    ffx::AdjListManager adj_list_manager;
    
    // Load T1
    auto metadata1 = ffx::read_metadata_binary(t1_dir);
    std::vector<std::string> t1_cols;
    for (const auto& col : metadata1.columns) t1_cols.push_back(col.attr_name);
    
    ffx::LoadedTable t1 = ffx::load_table_from_columns_auto(
        t1_dir, "T1", t1_cols, query, adj_list_manager
    );
    
    // Load T2
    auto metadata2 = ffx::read_metadata_binary(t2_dir);
    std::vector<std::string> t2_cols;
    for (const auto& col : metadata2.columns) t2_cols.push_back(col.attr_name);
    
    ffx::LoadedTable t2 = ffx::load_table_from_columns_auto(
        t2_dir, "T2", t2_cols, query, adj_list_manager
    );
    
    // Verify both tables loaded
    EXPECT_EQ(t1.num_rows, 3);
    EXPECT_EQ(t2.num_rows, 3);
    
    // Verify NULL in T2.company
    const uint64_t* company_ids = t2.get_string_id_column("company");
    ASSERT_NE(company_ids, nullptr);
    EXPECT_EQ(company_ids[1], UINT64_MAX) << "Bob should have NULL company";
    
    // Verify adjacency lists
    EXPECT_TRUE(adj_list_manager.is_loaded(t1_dir, "id", "name", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(t2_dir, "name", "company", true));
    
    // Populate schema
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    ffx::populate_schema_from_adj_list_manager(
        schema, query, t1_dir, "id", "name", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, t2_dir, "name", "company", adj_list_manager
    );
    
    EXPECT_TRUE(schema.has_adj_list("id", "name"));
    EXPECT_TRUE(schema.has_adj_list("name", "company"));
    
    // Cleanup
    if (fs::exists(t1_dir)) {
        fs::remove_all(t1_dir);
    }
    if (fs::exists(t2_dir)) {
        fs::remove_all(t2_dir);
    }
}

TEST_F(DatalogE2ETest, StringToStringJoinAllNulls) {
    // Test edge case: string-to-string join where all destination values are NULL
    // Table T1: Person(name, friend_name)
    // name: string, friend_name: string (all NULLs)
    
    std::string table5_dir = fs::temp_directory_path() / "datalog_e2e_table5_all_nulls";
    fs::create_directories(table5_dir);
    
    std::vector<std::string> name = {"Alice", "Bob", "Charlie"};
    std::vector<bool> name_null = {false, false, false};
    std::vector<std::string> friend_name = {"", "", ""};  // All NULL
    std::vector<bool> friend_name_null = {true, true, true};
    
    ffx::serialize_string_column(name, name_null, table5_dir + "/name_string.bin");
    ffx::serialize_string_column(friend_name, friend_name_null, table5_dir + "/friend_name_string.bin");
    
    std::vector<ffx::ColumnConfig> configs;
    configs.emplace_back("name", 0, ffx::ColumnType::STRING);
    configs.emplace_back("friend_name", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values = {0, 0};
    ffx::write_metadata_binary(table5_dir, configs, 3, max_values);
    
    // Query: T5(n, f)
    ffx::Query query{"Q(n, f) := T5(n, f)"};
    
    ffx::AdjListManager adj_list_manager;
    
    auto metadata = ffx::read_metadata_binary(table5_dir);
    std::vector<std::string> t5_cols;
    for (const auto& col : metadata.columns) t5_cols.push_back(col.attr_name);
    
    // Load table
    ffx::LoadedTable t5 = ffx::load_table_from_columns_auto(
        table5_dir, "T5", t5_cols, query, adj_list_manager
    );
    
    EXPECT_EQ(t5.num_rows, 3);
    
    // Verify all friend_name values are NULL
    const uint64_t* friend_name_ids = t5.get_string_id_column("friend_name");
    ASSERT_NE(friend_name_ids, nullptr);
    for (uint64_t i = 0; i < t5.num_rows; i++) {
        EXPECT_EQ(friend_name_ids[i], UINT64_MAX) << "All friend_name values should be NULL";
    }
    
    // Verify adjacency lists - should be empty since all edges have NULL destination
    EXPECT_TRUE(adj_list_manager.is_loaded(table5_dir, "name", "friend_name", true));
    
    auto* fwd_adj = adj_list_manager.get_or_load(table5_dir, "name", "friend_name", true);
    ASSERT_NE(fwd_adj, nullptr);
    
    // All adjacency lists should be empty (no valid edges)
    // We can't easily check this without knowing the dictionary size, but we verify it doesn't crash
    
    // Populate schema
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    ffx::populate_schema_from_adj_list_manager(
        schema, query, table5_dir, "name", "friend_name", adj_list_manager
    );
    
    // Cleanup
    if (fs::exists(table5_dir)) {
        fs::remove_all(table5_dir);
    }
}

TEST_F(DatalogE2ETest, ThreeTableJoinWithStringNulls) {
    // Test three-table join with string columns containing NULLs
    // T1: Person(id, name) - id: uint64, name: string
    // T2: WorksAt(name, company) - name: string, company: string (some NULLs)
    // T3: LocatedIn(company, city) - company: string, city: string (some NULLs)
    // Join chain: T1.name = T2.name, T2.company = T3.company
    
    std::string t1_dir = fs::temp_directory_path() / "datalog_e2e_3t1_nulls";
    std::string t2_dir = fs::temp_directory_path() / "datalog_e2e_3t2_nulls";
    std::string t3_dir = fs::temp_directory_path() / "datalog_e2e_3t3_nulls";
    fs::create_directories(t1_dir);
    fs::create_directories(t2_dir);
    fs::create_directories(t3_dir);
    
    // T1: Person(id, name)
    std::vector<uint64_t> id = {0, 1, 2};
    std::vector<std::string> name = {"Alice", "Bob", "Charlie"};
    std::vector<bool> name_null = {false, false, false};
    
    uint64_t max_id = 2;
    ffx::serialize_uint64_column(id, t1_dir + "/id_uint64.bin", &max_id);
    ffx::serialize_string_column(name, name_null, t1_dir + "/name_string.bin");
    
    std::vector<ffx::ColumnConfig> configs1;
    configs1.emplace_back("id", 0, ffx::ColumnType::UINT64);
    configs1.emplace_back("name", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values1 = {max_id, 0};
    ffx::write_metadata_binary(t1_dir, configs1, 3, max_values1);
    
    // T2: WorksAt(name, company) - company has NULLs
    std::vector<std::string> name2 = {"Alice", "Bob", "Charlie"};
    std::vector<bool> name2_null = {false, false, false};
    std::vector<std::string> company = {"Google", "", "Microsoft"};  // Bob has NULL
    std::vector<bool> company_null = {false, true, false};
    
    ffx::serialize_string_column(name2, name2_null, t2_dir + "/name_string.bin");
    ffx::serialize_string_column(company, company_null, t2_dir + "/company_string.bin");
    
    std::vector<ffx::ColumnConfig> configs2;
    configs2.emplace_back("name", 0, ffx::ColumnType::STRING);
    configs2.emplace_back("company", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values2 = {0, 0};
    ffx::write_metadata_binary(t2_dir, configs2, 3, max_values2);
    
    // T3: LocatedIn(company, city) - city has NULLs
    std::vector<std::string> company3 = {"Google", "Microsoft", "Apple"};
    std::vector<bool> company3_null = {false, false, false};
    std::vector<std::string> city = {"Mountain View", "", "Cupertino"};  // Microsoft city is NULL
    std::vector<bool> city_null = {false, true, false};
    
    ffx::serialize_string_column(company3, company3_null, t3_dir + "/company_string.bin");
    ffx::serialize_string_column(city, city_null, t3_dir + "/city_string.bin");
    
    std::vector<ffx::ColumnConfig> configs3;
    configs3.emplace_back("company", 0, ffx::ColumnType::STRING);
    configs3.emplace_back("city", 1, ffx::ColumnType::STRING);
    std::vector<uint64_t> max_values3 = {0, 0};
    ffx::write_metadata_binary(t3_dir, configs3, 3, max_values3);
    
    // Query: T1(i, n), T2(n, c), T3(c, city)
    ffx::Query query{"Q(i, n, c, city) := T1(i, n), T2(n, c), T3(c, city)"};
    
    ffx::AdjListManager adj_list_manager;
    
    // Load all three tables
    auto metadata1 = ffx::read_metadata_binary(t1_dir);
    std::vector<std::string> t1_cols;
    for (const auto& col : metadata1.columns) t1_cols.push_back(col.attr_name);
    
    auto metadata2 = ffx::read_metadata_binary(t2_dir);
    std::vector<std::string> t2_cols;
    for (const auto& col : metadata2.columns) t2_cols.push_back(col.attr_name);
    
    auto metadata3 = ffx::read_metadata_binary(t3_dir);
    std::vector<std::string> t3_cols;
    for (const auto& col : metadata3.columns) t3_cols.push_back(col.attr_name);
    
    ffx::LoadedTable t1 = ffx::load_table_from_columns_auto(
        t1_dir, "T1", t1_cols, query, adj_list_manager
    );
    ffx::LoadedTable t2 = ffx::load_table_from_columns_auto(
        t2_dir, "T2", t2_cols, query, adj_list_manager
    );
    ffx::LoadedTable t3 = ffx::load_table_from_columns_auto(
        t3_dir, "T3", t3_cols, query, adj_list_manager
    );
    
    // Verify NULLs
    const uint64_t* t2_company_ids = t2.get_string_id_column("company");
    ASSERT_NE(t2_company_ids, nullptr);
    EXPECT_EQ(t2_company_ids[1], UINT64_MAX) << "Bob should have NULL company";
    
    const uint64_t* t3_city_ids = t3.get_string_id_column("city");
    ASSERT_NE(t3_city_ids, nullptr);
    EXPECT_EQ(t3_city_ids[1], UINT64_MAX) << "Microsoft should have NULL city";
    
    // Verify adjacency lists
    EXPECT_TRUE(adj_list_manager.is_loaded(t1_dir, "id", "name", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(t2_dir, "name", "company", true));
    EXPECT_TRUE(adj_list_manager.is_loaded(t3_dir, "company", "city", true));
    
    // Populate schema
    ffx::Schema schema;
    schema.adj_list_manager = &adj_list_manager;
    
    ffx::populate_schema_from_adj_list_manager(
        schema, query, t1_dir, "id", "name", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, t2_dir, "name", "company", adj_list_manager
    );
    ffx::populate_schema_from_adj_list_manager(
        schema, query, t3_dir, "company", "city", adj_list_manager
    );
    
    EXPECT_TRUE(schema.has_adj_list("id", "name"));
    EXPECT_TRUE(schema.has_adj_list("name", "company"));
    EXPECT_TRUE(schema.has_adj_list("company", "city"));
    
    // Cleanup
    if (fs::exists(t1_dir)) {
        fs::remove_all(t1_dir);
    }
    if (fs::exists(t2_dir)) {
        fs::remove_all(t2_dir);
    }
    if (fs::exists(t3_dir)) {
        fs::remove_all(t3_dir);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

