#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help="CMake build directory containing FFX binaries (or set FFX_BUILD_DIR).",
    )
    ap.add_argument("--provider", default="demo")
    ap.add_argument("--model", default="demo")
    ap.add_argument("--output", default="output/ffx_keywords.csv")
    ap.add_argument("--tuple-format", default="MARKDOWN", choices=["MARKDOWN", "JSON", "FTREE"])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    root_dir = pathlib.Path(__file__).parent.resolve()
    out_path = (root_dir / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    query_exec = _require_binary(_resolve_build_dir(args), "query_eval_exec")
    data_root = root_dir / "serialized"
    if not data_root.exists():
        raise FileNotFoundError("Run setup_data.py first to create examples/semantic/serialized/")
    ordering_str = (root_dir / "ordering.txt").read_text(encoding="utf-8").strip()

    prompt = (
        "TASK:\n"
        "For each triple (a,b,c) in the table rows, generate exactly 5 keywords or short keyphrases.\n"
        "Use ONLY the abstracts of a, b, and c (aabs, babs, cabs) to derive the keywords.\n\n"
        "OUTPUT FORMAT:\n"
        "Return one line per row with exactly 5 comma-separated keywords.\n"
        "No extra text, no numbering, no JSON, no quotes.\n"
    )

    # FTREE format expects context_column as {attr: capacity}, flat formats need a list.
    if args.tuple_format == "FTREE":
        context_column = {"aabs": 5, "babs": 5, "cabs": 5, "a": 8, "b": 8, "c": 8}
    else:
        context_column = ["aabs", "babs", "cabs", "a", "b", "c"]

    llm_cfg = {
        "provider": args.provider,
        "model": args.model,
        "model_name": "ffx_semantic_keywords",
        "tuple_format": args.tuple_format,
        "batch_size": args.batch_size,
        "context_column": context_column,
        "prompt": prompt,
    }
    llm_json = json.dumps(llm_cfg)

    # Head lists outputs; LLM_MAP binding is appended in the body.
    base_body = (root_dir / "query.txt").read_text(encoding="utf-8").strip()
    query_str = (
        "Q(a,atitle,aabs,b,btitle,babs,c,ctitle,cabs,keywords) := "
        f"{base_body}, keywords = LLM_MAP({llm_json})"
    )

    cmd = [
        str(query_exec),
        str(data_root),
        query_str,
        ordering_str,
        f"sink_csv:{str(out_path)}",
    ]
    print("Running command:\n  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(root_dir))
    shutil.rmtree(data_root)
    print(f"Removed {data_root}", flush=True)


if __name__ == "__main__":
    main()
