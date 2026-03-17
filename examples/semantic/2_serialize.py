#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess


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


def run_serializer(serializer: pathlib.Path, input_csv: pathlib.Path, out_dir: pathlib.Path, column_config: str) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / "column_config.txt"
    cfg.write_text(column_config, encoding="utf-8")
    subprocess.run([str(serializer), str(input_csv), str(out_dir), str(cfg)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serialize work/*.csv with table_serializer")
    ap.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help="CMake build directory containing FFX binaries (or set FFX_BUILD_DIR).",
    )
    args = ap.parse_args()
    serializer = _require_binary(_resolve_build_dir(args), "table_serializer")

    root_dir = pathlib.Path(__file__).parent.resolve()
    work_dir = root_dir / "work"
    ser_dir = root_dir / "serialized"
    ser_dir.mkdir(exist_ok=True)

    papers = work_dir / "papers.csv"
    edges = work_dir / "paper_edges.csv"
    if not papers.exists() or not edges.exists():
        raise FileNotFoundError("Run 1_normalize.py first to create examples/semantic/work/*.csv")

    run_serializer(
        serializer,
        papers,
        ser_dir / "papers",
        "\n".join([
            "COLUMN,pid,0,uint64",
            "COLUMN,title,1,string",
            "COLUMN,abstract,2,string",
            "PROJECTION,ATitle,0,1,1:1",
            "PROJECTION,AAbs,0,2,1:1",
            "PROJECTION,BTitle,0,1,1:1",
            "PROJECTION,BAbs,0,2,1:1",
            "PROJECTION,CTitle,0,1,1:1",
            "PROJECTION,CAbs,0,2,1:1",
            "",
        ]),
    )
    run_serializer(
        serializer,
        edges,
        ser_dir / "paper_edges",
        "\n".join([
            "COLUMN,src,0,uint64",
            "COLUMN,dst,1,uint64",
            "PROJECTION,AB,0,1,m:n",
            "PROJECTION,BC,0,1,m:n",
            "PROJECTION,R,0,1,m:n",
            "",
        ]),
    )

    print(f"Serialized files saved to: {ser_dir}")


if __name__ == "__main__":
    main()
