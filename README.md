# Diffusion Inpainting

## How to run

For better performance with huggingface API, set in `.env` file `HF_TOKEN=<huggingface token>`

Environment provided in `environment.yml` is suitable for lambda cluster.

To run pipeline:
`python PipelineTester.py <flags>`

| flag         | default value | type                | description           | options		         |
|--------------|---------------|---------------------|-----------------------|-----------------------|
| `--pipeline` | `vanilla`     | str                 | The pipeline to run   | `vanila` `TDPaint` `MaskBlur` `BackgroundCopy` `BackgroundReconstruction` `CopyAndBlur`   |
| `--src`      | `./media`     | str                 | Source media dir      | directory      |
| `--dst`      | `./results`   | str                 | Destination media dir | directory |
| `--evaluate`   | `False`             | boolean             | Use to run evaluation, results will appear in destination folder    | -    |
| `--interactive`   | `False`             | boolean             | Use to run interactive mode    | -    |
| `--skip-existing`   | `False`             | boolean             | Skip files that already have existing generation in destination dir    | -    |

To download dataset:
`python download_dataset.py --output_dir <dir> --samples_count <int>`

Use `invert.py` to invert all masks in a certain folder.

## Latent inpainting process visualization

![inpainting visualized](generation.gif)
