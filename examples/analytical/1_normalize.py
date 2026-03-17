import pathlib
from collections import defaultdict

def main():
    data_dir = pathlib.Path(__file__).parent.resolve() / "data"
    work_dir = pathlib.Path(__file__).parent.resolve() / "work"
    work_dir.mkdir(exist_ok=True)
    
    # 1. First pass: find all entity types and build id maps
    id_maps = defaultdict(dict)
    
    files = list(data_dir.glob("*.csv"))
    
    import re
    def get_types(header):
        parts = header.split("|")
        return [re.search(r'\((.*?)\)', p).group(1) for p in parts]
    
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
                        
    # 2. Second pass: map IDs and write to work_dir
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

    print(f"Normalized files saved to {work_dir}")

if __name__ == "__main__":
    main()
