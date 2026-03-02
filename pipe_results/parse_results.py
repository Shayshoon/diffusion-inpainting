import os
import csv
import ast
import re
import argparse
from pathlib import Path

def parse_file(file_path, pipeline_name):
    """Parses a single evaluation file and returns a flattened dictionary."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Locate the dictionary content 
    dict_start = content.find('{')
    dict_end = content.rfind('}') + 1
    if dict_start == -1:
        return None
        
    try:
        data = ast.literal_eval(content[dict_start:dict_end])
    except Exception as e:
        print(f"Error parsing dictionary in {file_path}: {e}")
        return None

    flattened_row = {'pipeline': pipeline_name}
    
    # Flattening logic: Category first in key name to ensure grouping 
    for metric_name, categories in data.items():
        for category_name, stats in categories.items():
            for stat_name, value in stats.items():
                # Format: category_metric_stat (e.g., masked_PSNR_mean)
                column_name = f"{category_name}_{metric_name}_{stat_name}"
                flattened_row[column_name] = value
                
    return flattened_row

def save_to_csv(rows, output_csv):
    """Writes the list of parsed rows to a CSV with grouped columns."""
    if not rows:
        return

    # Gather all unique keys across all rows
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    
    # Custom sorting to group categories: [pipeline, bbox_..., full_..., masked_..., unmasked_...]
    # Alphanumeric sort on 'category_metric_stat' naturally groups them 
    sorted_cols = sorted([k for k in all_keys if k != 'pipeline'])
    fieldnames = ['pipeline'] + sorted_cols

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Process evaluation results into a grouped CSV.")
    parser.add_argument("--file", help="Path to a specific file (e.g., eval.vanilla.txt)")
    parser.add_argument("--scan", action="store_true", help="Scan subdirectories for 'evaluation.txt'")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV name")
    args = parser.parse_args()

    results_to_save = []

    # Mode 1: Process a specific file
    if args.file:
        file_path = Path(args.file)
        # Extract pipeline name: 'eval.vanilla.txt' -> 'vanilla'
        pipeline_match = re.search(r'eval\.(.*?)\.txt', file_path.name)
        pipeline_name = pipeline_match.group(1) if pipeline_match else file_path.stem
        
        row = parse_file(file_path, pipeline_name)
        if row:
            results_to_save.append(row)

    # Mode 2: Scan subdirectories for evaluation.txt
    if args.scan:
        current_dir = Path('.')
        for subdir in current_dir.iterdir():
            if subdir.is_dir():
                eval_file = subdir / "evaluation.txt"
                if eval_file.exists():
                    print(f"Found results in: {subdir.name}")
                    row = parse_file(eval_file, subdir.name)
                    if row:
                        results_to_save.append(row)

    if results_to_save:
        save_to_csv(results_to_save, args.output)
        print(f"Done! Processed {len(results_to_save)} entries into {args.output}")
    else:
        print("No valid evaluation data found.")

if __name__ == "__main__":
    main()
