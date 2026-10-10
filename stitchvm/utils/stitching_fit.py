"""Closed-form stitching-layer fitting (Stage 1).

Fits a Conv2d stitching layer by affine least squares on unfolded patch
matrices: the ones column appended to X makes weight and bias the joint
optimum of ``min_{W,b} ||X W + 1 bᵀ - Y||²``.

Numerical note: ``torch.linalg.lstsq`` on CUDA uses the QR-based ``gels``
driver, which assumes a full-rank system (rank-deficient input is NOT
detected). Ensure the feature matrix has full column rank before fitting.

Usage:
    from stitchvm.utils.stitching_fit import fit_conv_svd

    # Z_batches: list of (C_in, H, W) tensors
    # Y_batches: list of (C_out, H', W') tensors
    conv = fit_conv_svd(conv, Z_batches, Y_batches)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Unfold utilities
# ---------------------------------------------------------------------------


def _conv_output_spatial(x_shape, conv, effective_padding=None):
    """Compute output spatial dimensions for a conv layer."""
    ndim = len(conv.kernel_size)
    spatial_in = x_shape[2:]  # skip N, C
    if effective_padding is None:
        effective_padding = conv.padding
    output = []
    for i in range(ndim):
        s_in = spatial_in[i]
        k = conv.kernel_size[i]
        s = conv.stride[i]
        p = effective_padding[i]
        d = conv.dilation[i]
        s_out = (s_in + 2 * p - d * (k - 1) - 1) // s + 1
        output.append(s_out)
    return tuple(output)


@torch.no_grad()
def unfold_to_patch_matrix(x, conv, dtype=torch.float64):
    """Unfold input tensor into a patch matrix for least-squares fitting.

    Respects the conv's padding_mode (e.g. 'replicate') so that the
    unfolded patches match what the conv actually sees during forward().

    Args:
        x: (1, C_in, H, W) tensor
        conv: nn.Conv2d
        dtype: output dtype

    Returns:
        X: (S, d) patch matrix where S = product of output spatial dims,
           d = C_in * product of kernel_size
        output_spatial: tuple of output spatial dimensions
    """
    if len(conv.kernel_size) != 2:
        raise ValueError(f"Only Conv2d is supported, got {type(conv)}")
    pad_mode = getattr(conv, "padding_mode", "zeros")

    # If the conv uses non-zero padding mode (e.g. 'replicate'), manually pad
    # the input first and then unfold with padding=0. This matches the conv's
    # actual forward() behavior: F.pad(input, ..., mode) then conv with pad=0.
    if pad_mode != "zeros":
        pad_amounts = []
        for p in reversed(conv.padding):
            pad_amounts.extend([p, p])
        x = F.pad(x, pad_amounts, mode=pad_mode)
        effective_padding = (0, 0)
    else:
        effective_padding = conv.padding

    patches = F.unfold(
        x.float(),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=effective_padding,
        dilation=conv.dilation,
    )
    # patches: (1, d, S)
    X = patches[0].T.to(dtype)  # (S, d)
    output_spatial = _conv_output_spatial(x.shape, conv, effective_padding)

    return X, output_spatial


# ---------------------------------------------------------------------------
# Patch matrix interpolation
# ---------------------------------------------------------------------------


@torch.no_grad()
def interpolate_patch_matrix(X, src_size, dst_size):
    """Interpolate patch matrix columns via bilinear F.interpolate.

    Args:
        X: (S, d) patch matrix where S = product(src_size)
        src_size: (H, W) spatial dims
        dst_size: (H', W') target spatial dims

    Returns:
        X_tilde: (M, d) where M = product(dst_size)
    """
    d = X.shape[1]
    X_nd = X.T.reshape(d, 1, *src_size)
    X_interp = F.interpolate(X_nd, size=dst_size, mode="bilinear", align_corners=True)
    return X_interp.reshape(d, -1).T


# ---------------------------------------------------------------------------
# Closed-form least squares
# ---------------------------------------------------------------------------


def _get_patch_matrix_for_sample(Z_b, conv, target_spatial, dtype):
    """Compute patch matrix X for one sample: unfold then interpolate to target."""
    X_b, output_spatial = unfold_to_patch_matrix(Z_b, conv, dtype)
    X_b = interpolate_patch_matrix(X_b, output_spatial, target_spatial)
    return X_b


@torch.no_grad()
def fit_conv_svd(
    conv: nn.Module,
    Z_batches: list,
    Y_batches: list,
    dtype=torch.float64,
) -> nn.Module:
    """Fit conv weight and bias via closed-form affine least squares.

    Stacks all per-sample patch matrices into one system, appends a ones
    column so the bias is part of the solve (weight and bias are the joint
    affine optimum), and solves with torch.linalg.lstsq. See the module
    docstring for the full-rank requirement of the CUDA gels driver.

    Args:
        conv: nn.Conv2d (weights/bias will be set in-place)
        Z_batches: list of (C_in, H, W) input feature tensors (one per sample).
            H and W may differ between samples; each source is interpolated
            to its paired target grid before joining the least-squares system.
        Y_batches: list of (C_out, H', W') target feature tensors (one per sample)
        dtype: accumulation dtype

    Returns:
        conv with fitted weights and bias
    """
    if len(Z_batches) != len(Y_batches):
        raise ValueError(
            f"Z_batches and Y_batches must have equal length, "
            f"got {len(Z_batches)} vs {len(Y_batches)}"
        )

    device = conv.weight.device
    C_in = conv.in_channels
    C_out = conv.out_channels

    # Pass 1: collect all patch matrices and targets
    X_all = []
    Y_all = []

    for Z_b, Y_b in tqdm(
        zip(Z_batches, Y_batches), total=len(Z_batches), desc="lstsq fit (collecting)"
    ):
        Z_b = Z_b.to(device, non_blocking=True, dtype=dtype).unsqueeze(0)
        Y_b = Y_b.to(device, non_blocking=True, dtype=dtype)

        target_spatial = tuple(Y_b.shape[1:])
        Y_vec = Y_b.reshape(C_out, -1).T  # (S, C_out)

        X_b = _get_patch_matrix_for_sample(Z_b, conv, target_spatial, dtype)

        X_all.append(X_b)
        Y_all.append(Y_vec)

        del Z_b, Y_b

    X = torch.cat(X_all, dim=0)  # (N*S, d)
    Y = torch.cat(Y_all, dim=0)  # (N*S, C_out)
    del X_all, Y_all
    torch.cuda.empty_cache()

    # Append a ones column so the intercept is solved jointly with the weight.
    if conv.bias is not None:
        X = torch.cat(
            [X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1
        )

    n_rows, n_cols = X.shape
    logger.info(
        f"lstsq: X is ({n_rows}, {n_cols}), "
        f"Y is ({n_rows}, {C_out})"
    )

    solution = torch.linalg.lstsq(X, Y).solution  # (d[+1], C_out)

    if conv.bias is not None:
        W = solution[:-1]
        conv.bias.copy_(solution[-1].to(conv.weight.dtype))
    else:
        W = solution
    kernel_shape = tuple(conv.kernel_size)
    conv.weight.copy_(W.T.reshape(C_out, C_in, *kernel_shape).to(conv.weight.dtype))

    del X, Y, solution
    torch.cuda.empty_cache()

    return conv
