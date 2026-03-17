# Flock prompt_manager + model_manager (adapted for ffx, no DuckDB)
# Prefers system nlohmann_json + fmt (CMake config). Falls back to FetchContent if not found
# (many distros ship libfmt-dev without fmtConfig.cmake).

include(FetchContent)

find_package(nlohmann_json QUIET CONFIG)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package; fetching from GitHub")
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(fmt QUIET CONFIG)
if(NOT fmt_FOUND)
    message(STATUS "fmt not found via find_package; fetching from GitHub")
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.2.1
    )
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fmt)
endif()

find_package(CURL REQUIRED)

add_library(libflock STATIC
    prompt_manager/prompt_manager.cpp
    prompt_manager/repository.cpp
    secret_manager/secret_manager.cpp
    model_manager/model.cpp
    model_manager/providers/adapters/openai.cpp
    model_manager/providers/adapters/azure.cpp
    model_manager/providers/adapters/ollama.cpp
    model_manager/providers/adapters/anthropic.cpp
    model_manager/providers/adapters/demo.cpp
    model_manager/providers/handlers/wasm_http.cpp
)
target_include_directories(libflock PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(libflock PUBLIC
    nlohmann_json::nlohmann_json
    fmt::fmt-header-only
    CURL::libcurl
)
