import torch

def mask_bbox(binary_mask: torch.Tensor):
    """Return (row_min, row_max, col_min, col_max) or None."""
    # binary_mask: (1, 1, H, W)
    m = binary_mask.squeeze()  # (H, W)
    rows = torch.any(m > 0.5, dim=1)
    cols = torch.any(m > 0.5, dim=0)
    if not rows.any():
        return None
    r0, r1 = rows.nonzero(as_tuple=False)[[0, -1]].flatten().tolist()
    c0, c1 = cols.nonzero(as_tuple=False)[[0, -1]].flatten().tolist()
    return int(r0), int(r1) + 1, int(c0), int(c1) + 1

def extract_regions(src_tensor: torch.Tensor,
                    output_tensor: torch.Tensor,
                    binary_mask: torch.Tensor):
    """
    Returns a dict of (src_region, output_region, weight) tuples.
    All tensors are (1, C, H, W).  weight is a float or 1-d mask.

    Regions
    -------
    full      – entire image, no masking
    bbox      – tight bounding box around the mask (crop both images)
    masked    – pixels where binary_mask == 0
    unmasked  – pixels where binary_mask == 1
    """
    regions = {}

    # --- full image ---
    regions["full"] = (src_tensor, output_tensor, None)

    # --- bounding box + masked area ---
    bbox = mask_bbox(binary_mask)
    if bbox is None:
        regions["bbox"] = (src_tensor, output_tensor, None)
        regions["masked"] = (src_tensor, output_tensor, inv_mask)
    else:
        r0, r1, c0, c1 = bbox
        bounded_src = src_tensor[:, :, r0:r1, c0:c1] if src_tensor is not None else None
        bounded_output = output_tensor[:, :, r0:r1, c0:c1]
        inv_mask = 1.0 - binary_mask
        
        regions["bbox"] = (bounded_src, bounded_output, None)
        regions["masked"] = (bounded_src, bounded_output, inv_mask)

    # --- unmasked area ---
    regions["unmasked"] = (src_tensor, output_tensor, binary_mask)

    return regions
