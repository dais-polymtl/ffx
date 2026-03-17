import argparse
import os
import pathlib
import subprocess
import shutil

# Cardinality on PROJECTION (forward along stored src→dest, col0→col1)
_PROJECTION_CARD = {
    "Forum_containerOf_Post": "1:n",
    "City_isPartOf_Country": "n:1",
    "Person_isLocatedIn_City": "n:1",
    "Forum_hasMember_Person": "m:n",
    "Comment_replyOf_Post": "n:1",
    "Message_hasTag_Tag": "m:n",
    "Tag_hasType_TagClass": "n:1",
}


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
    ap = argparse.ArgumentParser(description="Serialize work/*.edges with table_serializer")
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

    for f in work_dir.glob("*.edges"):
        name = f.stem
        out_dir = ser_dir / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        card = _PROJECTION_CARD.get(name, "m:n")
        cfg = out_dir / "column_config.txt"
        cfg.write_text(
            "\n".join([
                "COLUMN,src,0,uint64",
                "COLUMN,dest,1,uint64",
                f"PROJECTION,{name},0,1,{card}",
                "",
            ]),
            encoding="utf-8",
        )

        subprocess.run([str(serializer), str(f), str(out_dir), str(cfg)], check=True)
    print(f"Serialized files saved to {ser_dir}")


if __name__ == "__main__":
    main()
