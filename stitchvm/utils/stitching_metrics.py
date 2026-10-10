"""Stitching-layer evaluation metric.

Only MSE is reported: it is the quantity the closed-form fit minimises, so
it is the most direct measure of how well a candidate stitching layer
reconstructs the target features.

All computation runs in float64 for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate_stitching_layer_spatial(
    stitching_layer: nn.Module,
    Z_batches: list,
    Y_batches: list,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Evaluate a fitted stitching layer with per-pixel MSE.

    Conv first, then interpolate to Y's spatial size. Samples are batched
    by source and target shape, so square and rectangular resolutions can
    share one evaluation. MSE is summed over all target elements before
    division; larger source grids do not receive extra weight.

    Args:
        stitching_layer: fitted Conv2d module.
        Z_batches:       list of (C_in, H, W) source features.
        Y_batches:       list of (C_out, H', W') target features.
        device:          torch device.
        batch_size:      number of samples per forward pass.

    Returns:
        {"mse": float}
    """
    if not isinstance(stitching_layer, nn.Conv2d):
        raise TypeError(f"Expected Conv2d, got {type(stitching_layer)}")
    if len(Z_batches) != len(Y_batches) or not Z_batches:
        raise ValueError("Source and target features must have equal, non-zero length")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    stitching_layer = stitching_layer.to(dtype=torch.float64, device=device)

    mse_sum = 0.0
    vox = 0

    shape_groups = {}
    for index, (source, target) in enumerate(zip(Z_batches, Y_batches)):
        shape = (tuple(source.shape), tuple(target.shape))
        shape_groups.setdefault(shape, []).append(index)
    batches = [
        indices[start : start + batch_size]
        for indices in shape_groups.values()
        for start in range(0, len(indices), batch_size)
    ]
    for indices in tqdm(batches):
        Z_batch = torch.stack([Z_batches[i] for i in indices]).to(
            device=device, dtype=torch.float64
        )
        Y_batch = torch.stack([Y_batches[i] for i in indices]).to(
            device=device, dtype=torch.float64
        )
        target_size = list(Y_batch.shape[2:])

        pred = stitching_layer(Z_batch)
        pred = F.interpolate(
            pred, size=target_size, mode="bilinear", align_corners=True
        )

        diff = pred - Y_batch
        mse_sum += diff.square().sum().item()
        vox += pred.numel()

        del Z_batch, Y_batch, pred, diff
        torch.cuda.empty_cache()

    stitching_layer = stitching_layer.to(dtype=torch.float32)

    return {"mse": mse_sum / vox}
