#!/usr/bin/env python3
import argparse
import csv
import os
import pathlib
import shutil
import subprocess
import yaml
import json

def _resolve_build_dir(args: argparse.Namespace) -> pathlib.Path:
    raw = args.build_dir
    if raw is not None:
        bd = pathlib.Path(raw).expanduser().resolve()
    else:
        env = os.environ.get("FFX_BUILD_DIR")
        if not env:
            raise SystemExit("Set CMake build directory: --build-dir DIR or export FFX_BUILD_DIR=DIR")
        bd = pathlib.Path(env).expanduser().resolve()
    if not bd.is_dir():
        raise SystemExit(f"Not a directory: {bd}")
    return bd

def _require_binary(build_dir: pathlib.Path, name: str) -> pathlib.Path:
    p = build_dir / name
    if not p.is_file():
        raise SystemExit(f"Missing {name!r} in {build_dir}")
    return p

def main():
    ap = argparse.ArgumentParser(description="Normalize and serialize semantic example data")
    ap.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help="CMake build directory containing FFX binaries (or set FFX_BUILD_DIR).",
    )
    args = ap.parse_args()
    
    root_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = root_dir / "data"
    work_dir = root_dir / "work"
    ser_dir = root_dir / "serialized"
    config_file = root_dir / "config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Missing {config_file}")
        
    with config_file.open("r") as f:
        config = yaml.safe_load(f)

    # 1. Normalize
    print("Step 1: Normalizing data...")
    work_dir.mkdir(exist_ok=True)
    
    papers_in = data_dir / "papers.csv"
    edges_in = data_dir / "paper_edges.csv"
    
    id_map: dict[str, int] = {}
    papers_rows: list[tuple[int, str, str]] = []
    with papers_in.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            raw_id, title, abstract = row[0].strip(), row[1].strip(), row[2].strip()
            if raw_id not in id_map:
                id_map[raw_id] = len(id_map)
            papers_rows.append((id_map[raw_id], title, abstract))

    edges_rows: list[tuple[int, int]] = []
    with edges_in.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            s, d = row[0].strip(), row[1].strip()
            if s not in id_map: id_map[s] = len(id_map)
            if d not in id_map: id_map[d] = len(id_map)
            edges_rows.append((id_map[s], id_map[d]))

    papers_out = work_dir / "papers.csv"
    edges_out = work_dir / "paper_edges.csv"
    with papers_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in papers_rows: w.writerow(r)
    with edges_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in edges_rows: w.writerow(r)
    
    with (work_dir / "id_map.json").open("w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, sort_keys=True)

    # 2. Serialize
    print("Step 2: Serializing data...")
    serializer = _require_binary(_resolve_build_dir(args), "table_serializer")
    ser_dir.mkdir(exist_ok=True)

    for table_name, table_conf in config.get("tables", {}).items():
        input_csv = root_dir / table_conf["source"]
        out_dir = ser_dir / table_name
        if out_dir.exists(): shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cfg_out = out_dir / "column_config.txt"
        cfg_out.write_text("\n".join(table_conf["column_config"]) + "\n", encoding="utf-8")
        subprocess.run([str(serializer), str(input_csv), str(out_dir), str(cfg_out)], check=True)

    print(f"Setup complete. Data ready in {ser_dir}")

if __name__ == "__main__":
    main()
