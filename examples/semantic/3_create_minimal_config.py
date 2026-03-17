#!/usr/bin/env python3
"""Compatibility helper for serialized-root mode in the minimal semantic example."""

from __future__ import annotations

import pathlib


def main() -> None:
    root_dir = pathlib.Path(__file__).parent.resolve()
    ser_dir = root_dir / "serialized"
    edges = ser_dir / "paper_edges"

    if not edges.exists():
        raise FileNotFoundError("Run 1_normalize.py and 2_serialize.py first")

    out_file = root_dir / "data_root_minimal.txt"
    out_file.write_text(str(ser_dir.resolve()) + "\n", encoding="utf-8")
    print("Serialized-root mode: no table_config_minimal.txt needed.")
    print(f"Minimal data root saved to: {out_file}")


if __name__ == "__main__":
    main()
