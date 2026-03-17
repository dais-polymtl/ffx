# Analytical example

Small end-to-end demo: tiny **edge CSVs** → dense IDs → **serialized columnar tables** → one **`query_eval_exec`** . Run `query_eval_exec` with no arguments to print query form examples.

## What the data is

Input files under `data/*.csv` are **tables**: each row is one edge. Headers look like `:START_ID(City)|:END_ID(Country)` — the types in parentheses name the two endpoints. `1_normalize.py` assigns **dense 0-based uint64 IDs** per entity type and writes `work/<Relation>.edges` as `src|dst` lines. `2_serialize.py` turns each `.edges` file into a directory under `serialized/<Relation>/` (uint64 `src`/`dest` columns plus metadata) using the **`table_serializer`** binary.

The query chains seven such relations (country → city → person → forum → post → comment → tag → tag class) and counts matching tuples.

## Prerequisites

Build the C++ tools from the repo root (see the main [README](../../README.md)) so these exist:

- `build/table_serializer` (or `DEBUG/` / `RELEASE/` if you use those build dirs)
- `build/query_eval_exec`

`4_run_query.py` searches `build/`, then `DEBUG/`, then `RELEASE/` for `query_eval_exec`.

## Pipeline

From **`examples/analytical/`** (or prefix paths with `examples/analytical/` if you run from the repo root):

| Step | Script | What it does |
|------|--------|----------------|
| 1 | `1_normalize.py` | Reads `data/*.csv`, writes `work/*.edges` with remapped IDs. |
| 2 | `2_serialize.py` | Runs `table_serializer` on each `work/*.edges` → `serialized/<name>/`. |
| 3 | `3_create_config.py` | Verifies `serialized/` and writes `data_root.txt` (absolute path to `serialized/`). The Python driver does not read this file; it always points at `serialized/` next to the scripts — the file is for your own tooling or docs. |
| 4 | `4_run_query.py` | Runs `query_eval_exec <serialized_dir> "<query from query.txt>" "<ordering from ordering.txt>"`. |

```bash
cd examples/analytical
python3 1_normalize.py
python3 2_serialize.py
python3 3_create_config.py
python3 4_run_query.py
```

## `query.txt` and `ordering.txt`

- **`query.txt`** — A single line in **rule form**: `Q(COUNT(*)) := <atoms>`. Each atom is a serialized **table name** with variables matching its projection, e.g. `City_isPartOf_Country(city,country)`. Commas separate atoms; variable names that repeat across atoms are **join keys**.

```markdown
- **`ordering.txt`** — The join order.
```

Edit those two files to experiment.

## Other files

- **`table_config.txt`** — Lists serialized table paths and optional cardinality hints; useful as documentation of which tables this example includes.
