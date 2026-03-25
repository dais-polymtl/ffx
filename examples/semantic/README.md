# Semantic example

Join plus an LLM step. `setup_data.py` normalizes and serializes papers and citations. `run_query.py` executes the query and saves results to `output/`.

---

### Prerequisites

1. Build the project from the repo root to get the `table_serializer` and `query_eval_exec` binaries.
2. `export FFX_BUILD_DIR=/path/to/your/build` or pass `--build-dir` to the scripts.

---

### Pipeline

```bash
cd examples/semantic
python3 setup_data.py
python3 run_query.py
```

To use a real LLM (e.g. OpenAI):
```bash
python3 run_query.py --provider openai --model gpt-4o-mini
```

---

### Files and Folders

| Path | Role |
|------|------|
| `data/` | Raw CSVs. |
| `work/` | Normalized CSVs with dense IDs. |
| `serialized/` | Area for serialized tables. |
| `config.yaml` | Defines serialization parameters. |
| `query.txt` | Relational join body. |
| `ordering.txt` | Join variable order. |
| `output/` | Result CSV location. |
