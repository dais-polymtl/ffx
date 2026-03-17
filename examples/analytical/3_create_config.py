import pathlib

def main():
    root_dir = pathlib.Path(__file__).parent.resolve()
    ser_dir = root_dir / "serialized"

    if not ser_dir.exists():
        raise FileNotFoundError("Run 2_serialize.py first to create examples/analytical/serialized/")

    out_file = root_dir / "data_root.txt"
    out_file.write_text(str(ser_dir.resolve()) + "\n", encoding="utf-8")
    print("Serialized-root mode: no table_config.txt needed.")
    print(f"Data root saved to {out_file}")

if __name__ == "__main__":
    main()
