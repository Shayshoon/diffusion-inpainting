#!/usr/bin/env python3
"""
Binarize all <name>.mask.jpg files in a directory.
Pixels > threshold → 255 (white), otherwise → 0 (black).
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def binarize_mask(path: Path, threshold: int = 127, inplace: bool = False) -> Path:
    img = Image.open(path).convert("L")  # grayscale
    arr = np.array(img)
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)
    result = Image.fromarray(arr)

    out_path = path if inplace else path.with_name(path.name.replace(".mask.jpg", ".mask_bin.jpg"))
    result.save(out_path, "JPEG", quality=100)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Binarize *.mask.jpg files in a directory.")
    parser.add_argument("directory", help="Path to the directory containing mask files")
    parser.add_argument("--threshold", type=int, default=127, help="Pixel threshold (default: 127)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original files")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    masks = sorted(directory.glob("*.mask.jpg"))
    if not masks:
        print("No *.mask.jpg files found.")
        return

    print(f"Found {len(masks)} mask(s). Threshold: {args.threshold}")
    for mask_path in masks:
        out = binarize_mask(mask_path, threshold=args.threshold, inplace=args.inplace)
        print(f"  {'[overwritten]' if args.inplace else '[saved]'} {out}")

    print("Done.")


if __name__ == "__main__":
    main()