import os
import re
import argparse
from tqdm import tqdm
from datasets import load_dataset


def sanitize(img_id: str) -> str:
    return re.sub(r'[^\w\-.]', '_', str(img_id))


def download_pipe_dataset(output_dir: str, samples_count: int = 250):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading datasets (n={samples_count})...")
    images = list(load_dataset('paint-by-inpaint/PIPE', split="test", streaming=True).take(samples_count))
    masks  = list(load_dataset('paint-by-inpaint/PIPE_Masks', split="test", streaming=True).take(samples_count))

    assert len(images) == len(masks), "Image/mask count mismatch"

    skipped = 0
    saved = 0

    for img_sample, mask_sample in tqdm(zip(images, masks), total=len(images), desc="Saving samples"):
        name = sanitize(img_sample['img_id'])

        src_path    = os.path.join(output_dir, f"{name}.jpg")
        mask_path   = os.path.join(output_dir, f"{name}.mask.jpg")
        prompt_path = os.path.join(output_dir, f"{name}.txt")

        # Skip if all three files already exist (resume interrupted download)
        if all(os.path.exists(p) for p in (src_path, mask_path, prompt_path)):
            skipped += 1
            continue

        img_sample['target_img'].convert('RGB').save(src_path, format="JPEG")
        mask_sample['mask'].convert('L').save(mask_path, format="JPEG")
        with open(prompt_path, "w") as f:
            prompt = (img_sample.get('Instruction_Class') or '') + (img_sample.get('Instruction_VLM-LLM') or '')
            f.write(prompt[4:])

        saved += 1

    print(f"Done. {saved} samples saved, {skipped} skipped (already existed) → '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save PIPE dataset samples to disk.")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save samples")
    parser.add_argument("--samples_count", type=int, default=250, help="Number of samples to download")
    args = parser.parse_args()

    download_pipe_dataset(args.output_dir, args.samples_count)

