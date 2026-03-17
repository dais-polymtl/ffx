<h1 align="center">FFX — Fast Factorized eXecution</h1>

<h2 align="center">A fast factorized query engine supporting semantic queries</h2>

<p align="center">
  <a href="#-what-is-ffx">What is FFX?</a> •
  <a href="#-requirements">Requirements</a> •
  <a href="#-getting-started-build--test">Build &amp; test</a> •
  <a href="#-run-an-example">Run an example</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-team">Team</a> •
  <a href="#-cite-this-paper">Cite</a>
</p>

---

<a id="readme-top"></a>

## 📖 What is FFX?

**FFX** is a C++ engine for join-heavy workloads: **factorized** intermediates plus **vectorized** execution. It runs standard analytical joins and filters, and can plug in an **LLM** step where the query asks for it.

## 📋 Requirements

| Tool | Version |
|------|---------|
| **CMake** | 3.5+ |
| **C++ compiler** | C++17 (GCC 7+ or Clang 15+) |
| **Libraries** | libcurl, nlohmann-json, fmt (dev packages) |
| **Python 3** | Optional, for example scripts |

## 🔧 Getting started (build & test)

Install **libcurl**, **nlohmann-json**, and **fmt** so CMake can `find_package` them.

**Debian / Ubuntu**

```bash
sudo apt update
sudo apt install -y build-essential cmake \
  libcurl4-openssl-dev nlohmann-json3-dev libfmt-dev
```

**macOS (Homebrew)** — `brew install cmake nlohmann-json fmt curl` (set `CMAKE_PREFIX_PATH` to `brew --prefix` if CMake cannot find packages).

**Fedora** — `sudo dnf install gcc-c++ cmake libcurl-devel nlohmann-json-devel fmt-devel`

**Windows** — same libraries via [vcpkg](https://vcpkg.io) or equivalents; pass `-DCMAKE_TOOLCHAIN_FILE=...` when needed.

Clone and build:

```bash
git clone https://github.com/dais-polymtl/ffx.git
cd ffx
cmake -S . -B RELEASE -DCMAKE_BUILD_TYPE=RELEASE
cmake --build RELEASE -j
```

Tests (optional):

```bash
cd RELEASE && ctest
```

## 🚀 Run an example

Pipeline: raw `data/` → `1_normalize.py` → `work/` → `2_serialize.py` (`table_serializer`) → `serialized/` → `query_eval_exec`. See **[`examples/analytical/README.md`](examples/analytical/README.md)** and **[`examples/semantic/README.md`](examples/semantic/README.md)**.

Semantic demo (after CMake build; set **`FFX_BUILD_DIR`** to that build directory):

```bash
cd examples/semantic
export FFX_BUILD_DIR=/path/to/your/cmake/build
python3 1_normalize.py && python3 2_serialize.py && python3 3_create_config.py && python3 4_run_query.py
```

Or invoke the engine directly:

```bash
/path/to/your/cmake/build/query_eval_exec examples/semantic/serialized \
  "Q(COUNT(*)) := T(x,y) WHERE x >= 0" "x,y"
```

CLI format: `query_eval_exec <serialized_root_dir> "<query>" "<ordering>"` plus optional flags (e.g. `-t` for threads).

## ✨ Team

Developed by the [**Data & AI Systems Laboratory (DAIS Lab)**](https://github.com/dais-polymtl) at **Polytechnique Montréal**.

## 📄 Cite this paper

```bibtex
@inproceedings{yasser2026ffx,
  author    = {Sunny Yasser and Anas Dorbani and Amine Mhedhbi},
  title     = {Factorized and Vectorized Execution: Optimizing Analytical and Semantic Queries over Relations},
  booktitle = {Proceedings of the ACM on Management of Data (SIGMOD)},
  volume    = {4},
  number    = {3},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3802055}
}
```

<p align="right"><a href="#readme-top">🔝 back to top</a></p>
