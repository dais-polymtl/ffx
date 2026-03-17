import argparse
import os
import pathlib
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


def main():
    ap = argparse.ArgumentParser(description="Run query_eval_exec on this example's serialized data")
    ap.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help="CMake build directory containing FFX binaries (or set FFX_BUILD_DIR).",
    )
    args = ap.parse_args()
    query_exec = _require_binary(_resolve_build_dir(args), "query_eval_exec")

    root_dir = pathlib.Path(__file__).parent.resolve()
    data_root = root_dir / "serialized"
    if not data_root.exists():
        raise FileNotFoundError("Run 2_serialize.py first to create examples/analytical/serialized/")

    query_file = root_dir / "query.txt"
    ordering_file = root_dir / "ordering.txt"
    query_str = query_file.read_text(encoding="utf-8").strip()
    ordering_str = ordering_file.read_text(encoding="utf-8").strip()

    cmd = [str(query_exec), str(data_root), query_str, ordering_str]
    print(f"Running command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
