import csv
import os

def generate_data(num_rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate papers.csv (pid, title, abstract)
    papers_path = os.path.join(output_dir, 'papers.csv')
    with open(papers_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(num_rows):
            title = f"Paper Title {i}"
            abstract = f"This is the abstract for paper {i}. It contains some text to simulate real data."
            writer.writerow([i, title, abstract])
            
    # Generate edges.csv (src, dst)
    # We'll create a chain 0->1, 1->2, ..., (N-1)->N
    edges_path = os.path.join(output_dir, 'edges.csv')
    with open(edges_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(num_rows - 1):
            writer.writerow([i, i + 1])

if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/integration_big/data"
    generate_data(3000, out_dir)
