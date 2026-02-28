# diffusion-inpainting


LatentPaint, TD-paint, Blended Latent Diffusion,\
post-conditioning, background reconstruction,\
KID, MSE

## TODO
- [] implement KID metric (with masked area cropped for better accuracy)
- [] implement TD-paint pipeline
- [] implement background reconstruction pipeline
- [] run metrics on dataset
- [] write report
- []
- []
- []

## Papers
The paper (Blended Latent Diffusion)[https://arxiv.org/pdf/2206.02779] introduces 2 techniques of interest. Background reconstruction and Progressive mask shrinking.
(LatentPaint)[https://openaccess.thecvf.com/content/WACV2024/papers/Corneanu_LatentPaint_Image_Inpainting_in_Latent_Space_With_Diffusion_Models_WACV_2024_paper.pdf] talks about conditioning. this is a method of making the model aware of the context that is the unmasked region throughout the diffusion process
(TD-Paint)[https://arxiv.org/pdf/2410.09306] basically says, the background doesnt need to be entirely noisy. we can noise the latent background using a low constant time step, like $t=50$, and the foreground like usual using the scheduler. then merge the bg and fg using mask, and continue diffusion process as usual.

## Dataset
(masks)[https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks/viewer/default/test]
(images)[https://huggingface.co/datasets/paint-by-inpaint/PIPE/viewer/default/test]

## Improvements
- background reconstruction
- progressive mask shrinking
- post conditioning

## Metrics
- MSE on background
    MSE on the background of a sample (unmasked area) is a simple metric that can illustrate the improvement on background preservation. This is because the background is not expected to change at all, therefore a pixel-level comparison should provide a reasonable metric.
- (regionCLIP)[https://arxiv.org/pdf/2112.09106], (git)[https://github.com/microsoft/RegionCLIP]
    To measure how closely the model followed the prompt, we can use precision. This is basically an 
- (KID)[https://arxiv.org/pdf/1801.01401], (docs)[https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html]
    Basically FID but works on smaller datasets

## Experiments
- Implemented a variation on TD-Paint. I didn't want to waste time and resources on training a model. I wanted a zero shot technique for inpainting. A promising idea was TD-Paint, basically I added very light noise to the background, hoping that it would help give context of the actual background to the inferred region. I got very poor results and now im dropping the idea of TD-Paint

## How to run

For better performance with huggingface API, set in `.env` file `HF_TOKEN=<huggingface token>`

To run pipeline:
`python PipelineTester.py <flags>`

| flag         | default value | type                | description           |
|--------------|---------------|---------------------|-----------------------|
| `--pipeline` | `vanilla`     | `vanilla\|TDPaint` | The pipeline to run   |
| `--src`      | `./media`     | str                 | Source media dir      |
| `--dst`      | `./results`   | str                 | Destination media dir |
| `--metric`   | -             | boolean             | Use to run metrics    |

