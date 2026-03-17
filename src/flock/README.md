# Flock prompt and model managers (ffx copy)

This directory contains a copy of the **prompt_manager** and **model_manager** from the flock project, adapted to build without DuckDB. Flock is the default and only LLM backend for the **Map** operator.

## Dependencies

Install (e.g. on macOS with Homebrew):
- `brew install nlohmann-json fmt`
- CURL is usually provided by the OS / Xcode.

Or with **vcpkg** (manifest mode from the repo root, where `vcpkg.json` lives):
- Configure with your vcpkg toolchain file, for example:
  - `cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake`
  - then `cmake --build build`

The **Map** operator uses flock's `PromptManager` and `Model` to build prompts and call the configured LLM (OpenAI, Ollama, Azure, Anthropic). Set environment variables for secrets, e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

## What was stubbed (no DuckDB)

- **core/config.hpp** – No DB connection; `get_global_storage_path()` returns `temp_directory_path()`.
- **core/common.hpp** – Standard library only.
- **secret_manager** – `GetSecret()` reads from environment (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_API_URL`, `AZURE_*`).
- **model_manager/model.cpp** – When model JSON does not contain full resolved details (`model`, `provider`, `secret`, `tuple_format`, `batch_size`), the code resolves from `model_name` + `provider` and fills `secret` via `SecretManager::GetSecret()`; it no longer queries DuckDB.
- **prompt_manager** – `CreatePromptDetails()` with `prompt_name` throws; use inline `"prompt"` in JSON.
- **metrics/manager.hpp** – No-op stubs for `UpdateTokens`, `IncrementApiCalls`, `AddApiDuration`.

## Layout

- `include/flock/` – Headers (prompt_manager, model_manager, core, secret_manager, metrics stubs).
- `prompt_manager/`, `model_manager/`, `secret_manager/` – Source files.
- Provider adapters (openai, azure, ollama, anthropic) and handlers (session, base_handler, url_handler) are included; HTTP uses libcurl.
