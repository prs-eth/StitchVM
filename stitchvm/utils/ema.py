"""Exponential moving average of trainable model parameters.

Tracks EMA for every parameter with ``requires_grad=True`` (in StitchVM Stage 2
that's the stitching layer + the reward-model encoder suffix). Frozen modules
(VAE, DiT, aesthetic MLP) are not tracked — at eval time, EMA weights are
swapped into the model's trainable slots and the frozen weights stay put.

Constant decay
--------------
The configured decay applies from step 0 (no warmup ramp).

Distributed setup
-----------------
After DDP gradient sync + optimizer step, every rank holds identical
trainable parameters, so each rank can update its own EMA independently
and the results stay consistent. We do NOT all-reduce EMA tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMAState:
    """Per-parameter exponential moving average (constant decay)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.step = 0
        # name -> EMA tensor (cloned, detached, on the parameter's device)
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Call once per optimizer step, after .step()."""
        self.step += 1
        d = self.decay
        for name, p in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            shadow.mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA tensors into the model's trainable slots in-place.

        Returns a backup dict of the original tensors so the caller can
        restore them after gathering / evaluating with EMA.
        """
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            backup[name] = p.data.clone()
            p.data.copy_(shadow)
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Undo a prior apply_to() using the returned backup dict."""
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {
            "step": int(self.step),
            "decay": float(self.decay),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.step = int(sd["step"])
        self.decay = float(sd["decay"])

        for name, tensor in sd["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))
