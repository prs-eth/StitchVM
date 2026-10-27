"""Generative pairwise judge with a Qwen3.5 reward checkpoint (UnifiedReward-2.0-qwen35).

Input: image + question + two answer texts; output: a generated verdict, parsed from text.
The original and the stitched model share everything except the vision branch:

- prompt: the UnifiedReward pair-rank image-understanding prompt, verbatim from
  github.com/CodeGoat24/UnifiedReward @ ff6f5202 (inference_qwen/image_understanding/
  qwen_pair_rank_image_understanding.py; identical text in
  benchmark_evaluation/Qwen_evaluation/test_qwen_vlrewardbench.py);
- chat template of the checkpoint with enable_thinking=False (the model card's deployment);
- image policy: the checkpoint's own processor applied once to the original image;
- decoding: greedy (do_sample=False), fixed max_new_tokens, repetition_penalty 1.0;
- answer order: the official script's random.seed(0), one random() draw per item in
  dataset order;
- parsing: the official substring rule plus a strict rule (exactly one verdict string).

The stitched model replaces model.model.visual with StitchedVisual, which rebuilds the
processor-resized image exactly from pixel_values, runs VAE -> noisy latent at a fixed
sigma and seed -> DiT prefix -> stitching layer -> head blocks [N:] + merger on the same
grid, and returns the merged tokens. Everything downstream (placeholders, 3D positions,
the language model) is untouched.
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from stitchvm.models.feedforward.qwen2vl_vision_head import qwen_tokens_to_spatial
from stitchvm.models.feedforward.qwen35_vision_head import qwen35_unpatchify
from stitchvm.utils.vlrb_protocol import (  # noqa: F401  (re-exported protocol pieces)
    ANSWER_1,
    ANSWER_2,
    PROMPT_SOURCE,
    arrange,
    build_prompt,
    category_by_source,
    category_official_script,
    official_order_draws,
    parse_verdict,
)

GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)


class Qwen35Judge:
    """Original-checkpoint judge; set .visual_override to swap the vision branch."""

    def __init__(self, model, processor, max_new_tokens: int = 4096):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.original_visual = model.model.visual

    def inputs(self, image, prompt_text: str):
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        return self.processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self.model.device)

    @torch.no_grad()
    def generate(self, inputs, visual: nn.Module | None = None) -> tuple[str, int]:
        self.model.model.visual = visual if visual is not None else self.original_visual
        self.model.model.rope_deltas = None
        try:
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **GREEDY)
        finally:
            self.model.model.visual = self.original_visual
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode([new], skip_special_tokens=True)[0], int(new.numel())

    @torch.no_grad()
    def last_logits(self, inputs, visual: nn.Module | None = None) -> torch.Tensor:
        self.model.model.visual = visual if visual is not None else self.original_visual
        self.model.model.rope_deltas = None
        try:
            return self.model(**inputs).logits[0, -1].float()
        finally:
            self.model.model.visual = self.original_visual


class _VisualShim(nn.Module):
    """Interface the language model expects from model.model.visual."""

    def __init__(self, reference_visual: nn.Module):
        super().__init__()
        self._dtype = reference_visual.dtype
        self.spatial_merge_size = reference_visual.spatial_merge_size

    @property
    def dtype(self):
        return self._dtype


class SplitTeacherVisual(_VisualShim):
    """Identity injection: original blocks [:N] -> spatial map -> Qwen35VisionHead(blocks [N:] +
    merger). Must reproduce the original merged tokens exactly (parity test)."""

    def __init__(self, visual: nn.Module, head: nn.Module, split_layer: int):
        super().__init__(visual)
        self.visual, self.head, self.split_layer = visual, head, split_layer

    def forward(self, pixel_values, grid_thw=None, **kwargs):
        captured = {}
        hook = self.visual.blocks[self.split_layer - 1].register_forward_hook(
            lambda m, a, o: captured.__setitem__("h", o))
        try:
            self.visual(pixel_values, grid_thw=grid_thw)
        finally:
            hook.remove()
        merged = []
        start = 0
        for t, h, w in grid_thw.tolist():
            n = t * h * w
            tokens = captured["h"][start:start + n].unsqueeze(0)
            spatial = qwen_tokens_to_spatial(tokens, t, h, w, self.spatial_merge_size)[:, :, 0]
            merged.append(self.head(spatial.to(next(self.head.parameters()).dtype))[0])
            start += n
        merged = torch.cat(merged).to(self.dtype)
        return BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=merged)


class StitchedVisual(_VisualShim):
    """Stitched vision branch: processor-resized image -> VAE -> noisy latent (fixed sigma and
    seed) -> DiT prefix -> stitching layer -> head on the same grid -> merged tokens."""

    def __init__(self, reference_visual, stitched_model, policy, prompt_embedding, sigma: float, seed: int = 0):
        super().__init__(reference_visual)
        self.stitched = stitched_model
        self.policy = policy
        self.prompt_embeds, self.pooled_embeds = prompt_embedding
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.last_vae_device = None  # per call: "cuda", "cuda_retry" or "cpu" (OOM fallback)

    def _vae_encode(self, x):
        """VAE encode with an out-of-memory fallback: the resident language model leaves little
        room for the fp32 encoder at the largest processor sizes (up to 16.7 MP). Retry once on
        the GPU after emptying the cache, then encode on the CPU in fp32 (same weights and
        precision); the device used is recorded per item."""
        try:
            self.last_vae_device = "cuda"
            return self.stitched.vae_encoder_forward(x)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        try:
            self.last_vae_device = "cuda_retry"
            return self.stitched.vae_encoder_forward(x)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        vae = self.stitched.diffusion_vae
        dev = x.device
        vae.to("cpu")
        try:
            self.last_vae_device = "cpu"
            latent = self.stitched.vae_encoder_forward(x.cpu()).to(dev)
        finally:
            vae.to(dev)
        return latent

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw=None, **kwargs):
        merged = []
        start = 0
        dev = pixel_values.device
        for row in grid_thw:
            t, h, w = [int(v) for v in row.tolist()]
            n = t * h * w
            image = qwen35_unpatchify(pixel_values[start:start + n], row, self.policy).clamp(0, 1)
            start += n
            latent = self._vae_encode((image.unsqueeze(0) * 2 - 1).float())
            gen = torch.Generator(device=dev).manual_seed(self.seed)
            eps = torch.randn(latent.shape, generator=gen, device=dev, dtype=latent.dtype)
            noisy = (1 - self.sigma) * latent + self.sigma * eps
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.stitched(
                    noisy_latent=noisy,
                    timestep=torch.full((1,), self.sigma * 1000, device=dev),
                    encoder_hidden_states=self.prompt_embeds,
                    pooled_projections=self.pooled_embeds,
                    train=False,
                    target_grid=(h, w),
                )
            merged.append(out[0])
        merged = torch.cat(merged).to(self.dtype)
        return BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=merged)
