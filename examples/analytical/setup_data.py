#!/usr/bin/env python3
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import yaml
from collections import defaultdict

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

def get_types(header):
    parts = header.split("|")
    return [re.search(r'\((.*?)\)', p).group(1) for p in parts]

def main():
    ap = argparse.ArgumentParser(description="Normalize and serialize analytical example data")
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
    id_maps = defaultdict(dict)
    files = list(data_dir.glob("*.csv"))
    file_types = {}
    
    for f in files:
        with f.open("r", encoding="utf-8") as fin:
            header = fin.readline().strip()
            file_types[f] = get_types(header)
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"): continue
                line = line.replace(",", "|").replace("\t", "|")
                vals = line.split("|")
                if len(vals) >= 2:
                    src_val, dst_val = vals[0].strip(), vals[1].strip()
                    src_type, dst_type = file_types[f][0], file_types[f][1]
                    if src_val not in id_maps[src_type]:
                        id_maps[src_type][src_val] = len(id_maps[src_type])
                    if dst_val not in id_maps[dst_type]:
                        id_maps[dst_type][dst_val] = len(id_maps[dst_type])

    for f in files:
        src_type, dst_type = file_types[f][0], file_types[f][1]
        out_file = work_dir / f"{f.stem}.edges"
        with f.open("r", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
            fin.readline() # Skip header
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"): continue
                line = line.replace(",", "|").replace("\t", "|")
                vals = line.split("|")
                if len(vals) >= 2:
                    src_val, dst_val = vals[0].strip(), vals[1].strip()
                    mapped_src = id_maps[src_type][src_val]
                    mapped_dst = id_maps[dst_type][dst_val]
                    fout.write(f"{mapped_src}|{mapped_dst}\n")

    # 2. Serialize
    print("Step 2: Serializing data...")
    serializer = _require_binary(_resolve_build_dir(args), "table_serializer")
    ser_dir.mkdir(exist_ok=True)

    for table_name, table_conf in config.get("tables", {}).items():
        input_csv = root_dir / table_conf["source"]
        out_dir = ser_dir / table_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        cfg_out = out_dir / "column_config.txt"
        cfg_out.write_text("\n".join(table_conf["column_config"]) + "\n", encoding="utf-8")
        subprocess.run([str(serializer), str(input_csv), str(out_dir), str(cfg_out)], check=True)

    print(f"Setup complete. Data ready in {ser_dir}")

if __name__ == "__main__":
    main()
