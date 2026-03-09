import os
import argparse
from PIL import Image, ImageOps
from tqdm import tqdm


def invert_masks(samples_dir: str, threshold: int = 128):
    mask_files = [f for f in os.listdir(samples_dir) if f.endswith('.mask.jpg')]

    for filename in tqdm(mask_files, desc="Inverting masks"):
        path = os.path.join(samples_dir, filename)
        mask = Image.open(path).convert('L')
        mask = ImageOps.invert(mask)
        mask = mask.point(lambda x: 255 if x >= threshold else 0, 'L')
        mask.save(path, format="JPEG")

    print(f"Done. {len(mask_files)} masks inverted in '{samples_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert and binarize all mask images in a directory.")
    parser.add_argument("--samples_dir", type=str, default="samples", help="Directory containing mask files")
    parser.add_argument("--threshold", type=int, default=128, help="Threshold for binarization (default: 128)")
    args = parser.parse_args()

    invert_masks(args.samples_dir, args.threshold)
