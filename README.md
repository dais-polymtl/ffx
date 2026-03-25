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
| **Python 3.7 or above** | Optional, for example scripts |

## 🔧 Getting started (build & test)

Install **libcurl**, **nlohmann-json**, and **fmt** so CMake can `find_package` them.

**Debian / Ubuntu**

```bash
sudo apt update
sudo apt install -y build-essential cmake \
  libcurl4-openssl-dev nlohmann-json3-dev libfmt-dev
```

**macOS (Homebrew)** — `brew install cmake nlohmann-json fmt curl` (set `CMAKE_PREFIX_PATH` to `brew --prefix` if CMake cannot find packages).

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

See **[`examples/analytical/`](examples/analytical/)** and **[`examples/semantic/`](examples/semantic/)** for end-to-end demos of analytical and semantic queries. Each folder contains a `setup_data.py` script to prepare the data and a `run_query.py` script to execute the engine.

Or invoke the engine directly:

```bash
/path/to/your/build/query_eval_exec <serialized_root_dir> "<query>" "<ordering>"
```

## ✨ Team

Developed by the [**Data & AI Systems Laboratory (DAIS Lab)**](https://github.com/dais-polymtl) at **Polytechnique Montréal**.

## 📄 Cite this paper

```bibtex
@article{DBLP:journals/pacmmod/YasserDM26,
  author    = {Sunny Yasser and Anas Dorbani and Amine Mhedhbi},
  title     = {Factorized and Vectorized Execution: Optimizing Analytical and Semantic Queries over Relations},
  journal   = {Proc. {ACM} Manag. Data},
  volume    = {4},
  number    = {3},
  year      = {2026},
  url       = {https://doi.org/10.1145/3802055},
  doi       = {10.1145/3802055}
}
```

<p align="right"><a href="#readme-top">🔝 back to top</a></p>
