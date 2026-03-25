# Analytical example

Small end-to-end demo: tiny **edge CSVs** → dense IDs → **serialized columnar tables** → one **`query_eval_exec`** . Run `query_eval_exec` with no arguments to print query form examples.

## What the data is

Input files are relationship CSVs. `setup_data.py` normalizes them to dense IDs and serializes them.

The query chains seven such relations (country → city → person → forum → post → comment → tag → tag class) and counts matching tuples.

## Prerequisites

Build the C++ tools from the repo root (see the main [README](../../README.md)) so these exist:

- `build/table_serializer` (or `DEBUG/` / `RELEASE/` if you use those build dirs)
- `build/query_eval_exec`

`run_query.py` searches `build/`, then `DEBUG/`, then `RELEASE/` for `query_eval_exec`.

## Pipeline

From **`examples/analytical/`** (or prefix paths with `examples/analytical/` if you run from the repo root):

| Step | Script | What it does |
|------|--------|----------------|
| 1 | `setup_data.py` | Normalizes data to dense IDs from 0, and serializes it using `config.yaml`. |
| 2 | `run_query.py` | Runs `query_eval_exec <serialized_dir> "<query from query.txt>" "<ordering from ordering.txt>"`. |

```bash
cd examples/analytical
python3 setup_data.py
python3 run_query.py
```

## `query.txt` and `ordering.txt`

- **`query.txt`** — A single line in **rule form**: `Q(COUNT(*)) := <atoms>`. Each atom is a serialized **table name** with variables matching its projection, e.g. `City_isPartOf_Country(city,country)`. Commas separate atoms; variable names that repeat across atoms are **join keys**.

- **`ordering.txt`** — The join order.

Edit those two files to experiment.

## Other files

- **`config.yaml`** — Defines table names, source paths, and column/projection configurations for the serializer.
