#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib


def main() -> None:
    root_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = root_dir / "data"
    work_dir = root_dir / "work"
    work_dir.mkdir(exist_ok=True)

    papers_in = data_dir / "papers.csv"
    edges_in = data_dir / "paper_edges.csv"
    if not papers_in.exists() or not edges_in.exists():
        raise FileNotFoundError(
            "Missing input files in examples/semantic/data/. "
            "Expected papers.csv and paper_edges.csv."
        )

    # Map original paper IDs to dense uint64 ids (string->int).
    id_map: dict[str, int] = {}
    papers_rows: list[tuple[int, str, str]] = []
    with papers_in.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f"papers.csv row must be id,title,abstract: {row}")
            raw_id, title, abstract = row[0].strip(), row[1].strip(), row[2].strip()
            if raw_id not in id_map:
                id_map[raw_id] = len(id_map)
            papers_rows.append((id_map[raw_id], title, abstract))

    # Normalize edges to the dense ids.
    edges_rows: list[tuple[int, int]] = []
    with edges_in.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"paper_edges.csv row must be src,dst: {row}")
            s, d = row[0].strip(), row[1].strip()
            if s not in id_map:
                id_map[s] = len(id_map)
            if d not in id_map:
                id_map[d] = len(id_map)
            edges_rows.append((id_map[s], id_map[d]))

    papers_out = work_dir / "papers.csv"
    edges_out = work_dir / "paper_edges.csv"
    id_map_out = work_dir / "id_map.json"

    with papers_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for pid, title, abstract in papers_rows:
            w.writerow([pid, title, abstract])

    with edges_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for s, d in edges_rows:
            w.writerow([s, d])

    # Save original->dense mapping for reference.
    with id_map_out.open("w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, sort_keys=True)

    print(f"Normalized files saved to: {work_dir}")


if __name__ == "__main__":
    main()

