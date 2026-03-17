#!/usr/bin/env python3
from __future__ import annotations

import pathlib


def main() -> None:
    root_dir = pathlib.Path(__file__).parent.resolve()
    ser_dir = root_dir / "serialized"
    papers = ser_dir / "papers"
    edges = ser_dir / "paper_edges"

    if not papers.exists() or not edges.exists():
        raise FileNotFoundError("Run 2_serialize.py first to create examples/semantic/serialized/")

    out_file = root_dir / "data_root.txt"
    out_file.write_text(str(ser_dir.resolve()) + "\n", encoding="utf-8")
    print("Serialized-root mode: no table_config.txt needed.")
    print(f"Data root saved to: {out_file}")


if __name__ == "__main__":
    main()

