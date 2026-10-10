"""Stage 1: closed-form stitching-layer search (StitchVM).

For each (DiT-block, reward-head-block) pair, fit one 1x1 convolutional
stitching layer by closed-form least squares, then report MSE on a held-out
split. Candidates are the first four reward-encoder blocks and the first
half of DiT blocks. Measured layer selections and their search scope
are listed in README.md.

  feedforward_model :  clip | dfn_clip | aesthetic | hpsv2
  diffusion_model   :  sd3.5 | sd3.5_large | flux_dev

The multi-resolution search invocations live in
``scripts/stage1_find_stitching_layer.sh``.

``aesthetic`` reuses the CLIP feature path (the aesthetic predictor is a
small MLP on top of CLIP ViT-L/14). This search needs only the vision encoder.
"""

import os

import torch
import torchvision.transforms as T
from loguru import logger
from tqdm import tqdm

from config.stitching_config.base import find_layer_stitching_argument
from stitchvm.utils.stitching_metrics import evaluate_stitching_layer_spatial
from stitchvm.utils.stitching_fit import fit_conv_svd

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Paper search space (Appendix C.1): the stitching point lives within the
# first 4 reward-head blocks, so only enc_blocks_1..4 are hooked and fitted.
NUM_HEAD_BLOCKS = 4
DIT_SEARCH_FRACTION = 0.5


def _dit_search_depth(total_blocks: int) -> int:
    """Keep the first half of blocks, rounded down (at least one)."""
    return max(1, int(total_blocks * DIT_SEARCH_FRACTION))


# ─────────────────────────────────────────────────────────────────────────────
# Head adapter — selects which reward-model backbone supplies target features
# ─────────────────────────────────────────────────────────────────────────────


class _Head:
    """Thin adapter wrapping a feedforward target model (CLIP / HPSv2 / DFN-CLIP).

    Encapsulates the three differences between heads:
      - which hook helper to call for per-block feature capture,
      - which forward signature the model uses (CLIP: `pixel_values=`,
        open_clip variants: `model.visual(...)`),
      - which normalization stats apply (read off the processor).
    """

    def __init__(self, kind: str, model, processor):
        self.kind = kind  # "clip" | "hpsv2" | "dfn_clip"
        self.model = model
        # Only prefix features are consumed. Later blocks cannot affect them,
        # so omit those blocks from the forward pass as well as from the grid.
        if kind == "clip":
            encoder = model.vision_model.encoder
            encoder.layers = encoder.layers[:NUM_HEAD_BLOCKS]
        else:
            transformer = model.visual.transformer
            transformer.resblocks = transformer.resblocks[:NUM_HEAD_BLOCKS]
        self._normalize = T.Normalize(mean=processor.image_mean, std=processor.image_std)

    def register_hooks(self, feature_store):
        if self.kind == "clip":
            from stitchvm.models.feedforward.clip_head import register_clip_feature_hooks
            return register_clip_feature_hooks(
                self.model, feature_store=feature_store, num_blocks=NUM_HEAD_BLOCKS
            )
        if self.kind == "hpsv2":
            from stitchvm.models.feedforward.hpsv2_head import register_hpsv2_feature_hooks
            return register_hpsv2_feature_hooks(
                self.model, feature_store=feature_store, num_blocks=NUM_HEAD_BLOCKS
            )
        if self.kind == "dfn_clip":
            from stitchvm.models.feedforward.dfn_clip_head import (
                register_dfn_clip_feature_hooks,
            )
            return register_dfn_clip_feature_hooks(
                self.model, feature_store=feature_store, num_blocks=NUM_HEAD_BLOCKS
            )
        raise ValueError(f"unknown head kind: {self.kind}")

    def forward(self, ff_images_unit):
        """Run the head forward with images in [0, 1] (hooks capture features)."""
        normed = self._normalize(ff_images_unit)
        if self.kind == "clip":
            self.model(pixel_values=normed)
        else:
            self.model.visual(normed)


# ─────────────────────────────────────────────────────────────────────────────
# DiT-side helpers — Flux and SD3 share enough structure that one body works
# ─────────────────────────────────────────────────────────────────────────────


def _register_dit_hooks(dit, is_flux, dit_features, dit_patch_hw):
    """Register `dit_blocks_{i}` hooks on the retained prefix; return handles.

    Flux numbers dual blocks 1..N then single blocks N+1..M so a single
    consecutive index can reference any depth.
    """

    def make_hook(name):
        def hook(_module, _input, output):
            if dit_patch_hw[0] is None:
                raise RuntimeError("dit_patch_hw not initialized — caller forgot")
            # Both SD3 JointTransformerBlock and Flux dual/single blocks
            # return (encoder_hidden_states, hidden_states).
            hidden_states = output[1]
            B, _N, D = hidden_states.shape
            spatial = hidden_states.transpose(1, 2).reshape(
                B, D, dit_patch_hw[0], dit_patch_hw[1]
            )
            spatial = spatial.detach().to(dtype=torch.float16)
            # bf16 activations above the fp16 range (65504) would silently
            # become inf and poison the closed-form fit for this block.
            if not torch.isfinite(spatial).all():
                raise RuntimeError(
                    f"non-finite values in {name} features after fp16 cast "
                    "(bf16 activation exceeded the fp16 range)"
                )
            for i in range(B):
                dit_features.setdefault(name, []).append(spatial[i].cpu())

        return hook

    handles = []
    dual_blocks = list(dit.transformer_blocks)
    for idx, block in enumerate(dual_blocks):
        handles.append(block.register_forward_hook(make_hook(f"dit_blocks_{idx + 1}")))
    if is_flux:
        for idx, block in enumerate(dit.single_transformer_blocks):
            handles.append(
                block.register_forward_hook(
                    make_hook(f"dit_blocks_{len(dual_blocks) + idx + 1}")
                )
            )
    return handles


def _dit_patch_hw(dit, is_flux, h_lat, w_lat):
    """Spatial grid the DiT operates on (Flux packs 2×2 latents inside the wrapper)."""
    if is_flux:
        return h_lat // 2, w_lat // 2
    return h_lat // dit.config.patch_size, w_lat // dit.config.patch_size


def _dit_forward(dit, is_flux, latents, sigma, prompt_embeds, pooled_embeds):
    """One DiT forward with rectified-flow noisy latents (z_sigma = (1-σ)·z + σ·ε)."""
    eps = torch.randn_like(latents, device="cuda", dtype=torch.float32)
    sigma_bchw = sigma[:, None, None, None]
    z_sigma = (1 - sigma_bchw) * latents + sigma_bchw * eps
    if is_flux:
        # FluxDiTStitched takes raw σ ∈ [0, 1] (it scales internally) and
        # expects spatial latents that match its dtype.
        dit_dtype = next(dit.parameters()).dtype
        dit(
            hidden_states=z_sigma.to(dit_dtype),
            timestep=sigma,
            encoder_hidden_states=prompt_embeds.to(dit_dtype),
            pooled_projections=pooled_embeds.to(dit_dtype),
        )
    else:
        # SD3 takes timestep in [0, 1000].
        dit(
            hidden_states=z_sigma,
            timestep=sigma * 1000,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unified feature extractor
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def extract_features(
    head: _Head,
    vae,
    data_loader,
    args,
    dit,
    prompt_embeds,
    pooled_embeds,
    is_flux: bool = False,
):
    """Extract source (DiT-block) and target (head) features for a dataset.

    Returns:
        source_features: ``{"dit_blocks_i": [...]}`` keyed by 1-based block index.
        target_features: ``{"enc_blocks_j": [...]}`` as produced by the head's
            registered hooks.

    All cached tensors are float16 on CPU. Source spatial dimensions may
    vary between batches; the hook grid is updated from each encoded latent.
    """
    target_features: dict = {}
    dit_features: dict = {}

    ts_low, ts_high = args.timestep_range

    target_handles = head.register_hooks(feature_store=target_features)

    dit_patch_hw = [None, None]
    dit_handles = _register_dit_hooks(dit, is_flux, dit_features, dit_patch_hw)

    try:
        for step, data in tqdm(
            enumerate(data_loader), total=args.iterations_for_feature_extraction
        ):
            if step >= args.iterations_for_feature_extraction:
                break

            images = data["vae_image_tensor"].to(vae.device)
            ff_images = data["feedforward_image_tensor"].to(vae.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # ── VAE encode (dataloaders emit [-1, 1] already) ──
                latents = vae.encode(images).latent_dist.sample()
                shift = getattr(vae.config, "shift_factor", 0.0)
                scale = getattr(vae.config, "scaling_factor", 1.0)
                latents = (latents - shift) * scale

                # ── DiT forward (fills dit_features via hooks) ──
                _, _, h_lat, w_lat = latents.shape
                dit_patch_hw[0], dit_patch_hw[1] = _dit_patch_hw(
                    dit, is_flux, h_lat, w_lat
                )
                sigma = (
                    torch.rand(latents.shape[:1], device="cuda", dtype=torch.float32)
                    * (ts_high - ts_low)
                ) + ts_low
                _dit_forward(dit, is_flux, latents, sigma, prompt_embeds, pooled_embeds)

                # ── Head forward (fills target_features via hooks) ──
                head.forward((ff_images + 1) / 2)
    finally:
        for h in target_handles:
            h.remove()
        for h in dit_handles:
            h.remove()

    return dit_features, target_features


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers (factor the duplicated branches out of main)
# ─────────────────────────────────────────────────────────────────────────────


def _load_image_vae(args, device, is_flux):
    if is_flux:
        from stitchvm.utils.utils_for_thirdparty import load_flux_vae
        return load_flux_vae(args, device)
    from stitchvm.utils.utils_for_thirdparty import load_sd3_vae
    return load_sd3_vae(args, device)


def _load_dit_and_embeddings(args, device, is_flux):
    """Load the DiT backbone and its empty-prompt embeddings."""
    if is_flux:
        from stitchvm.utils.utils_for_thirdparty import (
            load_flux_dit,
            load_flux_prompt_embedding,
        )
        from stitchvm.models.backbones.flux import FluxDiTStitched

        flux_dit_full = load_flux_dit(args, device)
        total_blocks = len(flux_dit_full.transformer_blocks) + len(
            flux_dit_full.single_transformer_blocks
        )
        # Number dual-stream and single-stream blocks as one sequence; retain
        # only its first half. The wrapper handles packing and guidance.
        search_depth = _dit_search_depth(total_blocks)
        dit = FluxDiTStitched(
            flux_dit_full, search_depth, guidance_scale=args.flux_guidance_scale
        )
        del flux_dit_full
        dit.to(device).eval()
        prompt_embeds, pooled_embeds = load_flux_prompt_embedding(args, device)
        return dit, prompt_embeds, pooled_embeds

    from stitchvm.utils.utils_for_thirdparty import (
        load_sd3_dit,
        load_sd3_prompt_embedding,
    )

    dit = load_sd3_dit(args, device)
    total_blocks = len(dit.transformer_blocks)
    search_depth = _dit_search_depth(total_blocks)
    dit.transformer_blocks = dit.transformer_blocks[:search_depth]
    logger.info(f"Searching the first {search_depth}/{total_blocks} SD3.5 DiT blocks")
    prompt_embeds, pooled_embeds = load_sd3_prompt_embedding(args, device)
    return dit, prompt_embeds, pooled_embeds


def _build_head(args, device) -> _Head:
    """Build the head specified by ``args.feedforward_model``."""
    ff = args.feedforward_model
    if ff == "dfn_clip":
        from stitchvm.utils.utils_for_thirdparty import load_dfn_clip_model
        model, processor = load_dfn_clip_model(args, device)
        return _Head("dfn_clip", model, processor)
    if ff == "hpsv2":
        from stitchvm.utils.utils_for_thirdparty import load_hpsv2_model
        model, processor = load_hpsv2_model(args, device)
        return _Head("hpsv2", model, processor)
    if ff in ("clip", "aesthetic"):
        # The aesthetic predictor is a frozen MLP on top of CLIP ViT-L/14
        # image embeddings, so Stage 1 only needs the CLIP feature pyramid;
        # the MLP is not needed for layer search.
        if ff == "aesthetic":
            args.clip_model_id = "openai/clip-vit-large-patch14"
            logger.info("Aesthetic mode: using openai/clip-vit-large-patch14 for layer search")
        from stitchvm.utils.utils_for_thirdparty import load_clip_model
        model, processor = load_clip_model(args, device)
        return _Head("clip", model, processor)
    raise ValueError(
        f"Unsupported feedforward_model: {ff}. "
        "Supported: clip, dfn_clip, aesthetic, hpsv2."
    )


def _diffusion_tag(args) -> str:
    return {"flux_dev": "flux_dev", "sd3.5_large": "sd3_large"}.get(
        args.diffusion_model, "sd3"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(args):
    device = "cuda"
    is_flux = args.diffusion_model == "flux_dev"

    os.makedirs(args.feature_save_path, exist_ok=True)

    ts_low, ts_high = args.timestep_range
    ff_tag = args.feedforward_model
    diff_tag = _diffusion_tag(args)
    tag = f"{ff_tag}_{diff_tag}_dit_t{ts_low:.2f}_{ts_high:.2f}"
    feature_file = os.path.join(args.feature_save_path, f"features_{tag}.pt")

    # ── Extract features (or load cached) ──
    if os.path.exists(feature_file):
        logger.info(f"Loading cached features from {feature_file}")
        cached = torch.load(feature_file, map_location="cpu", weights_only=False)
        source_features = cached["source_features"]
        target_features = cached["target_features"]
    else:
        logger.info("Extracting features...")
        from stitchvm.data.dataset_util import create_stitching_dataloader

        data_loader, _ = create_stitching_dataloader(
            args.dataset, args, augmentation=False
        )
        image_vae = _load_image_vae(args, device, is_flux)
        head = _build_head(args, device)
        dit, prompt_embeds, pooled_embeds = _load_dit_and_embeddings(
            args, device, is_flux
        )
        source_features, target_features = extract_features(
            head=head,
            vae=image_vae,
            data_loader=data_loader,
            args=args,
            dit=dit,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            is_flux=is_flux,
        )
        torch.save(
            {"source_features": source_features, "target_features": target_features},
            feature_file,
        )
        logger.info(f"Features saved at {feature_file}")

    torch.cuda.empty_cache()

    for k, v in source_features.items():
        shapes = sorted({tuple(sample.shape) for sample in v})
        logger.info(f"Source {k}: {len(v)} samples, shapes={shapes}")
    for k, v in target_features.items():
        logger.info(f"Target {k}: {len(v)} samples, shape={v[0].shape}")

    # ── Fit one stitching layer per (src, tgt) pair, eval on held-out split ──
    val_ratio = args.val_ratio

    # Sort by the trailing integer ("enc_blocks_10" must come after
    # "enc_blocks_2", not before — Python's lexical sort gets this wrong).
    def _natural_key(name: str):
        import re
        m = re.search(r"(\d+)$", name)
        return (name[: m.start()] if m else name, int(m.group(1)) if m else 0)

    tgt_keys_sorted = sorted(target_features.keys(), key=_natural_key)

    for tgt_key in tgt_keys_sorted:
        for src_key in sorted(source_features.keys(), key=_natural_key):
            src_list = source_features[src_key]
            tgt_list = target_features[tgt_key]
            in_channels = src_list[0].shape[0]

            n_total = len(src_list)
            n_val = max(1, int(n_total * val_ratio))
            n_train = n_total - n_val
            src_train, src_val = src_list[:n_train], src_list[n_train:]
            tgt_train, tgt_val = tgt_list[:n_train], tgt_list[n_train:]
            logger.info(f"Split: {n_train} train, {n_val} val (total {n_total})")

            conv = args.stitching_layer_config.build(in_channels=in_channels).to(device)

            combo = f"{src_key}_{tgt_key}"
            # The "_post" suffix is part of the published filename contract:
            # downstream consumers load state_dict_<src>_<tgt>_post.pt.
            state_path = os.path.join(
                args.feature_save_path, f"state_dict_{combo}_post.pt",
            )

            if os.path.exists(state_path):
                logger.info(f"Loading existing state dict for {combo}")
                sd = torch.load(state_path, map_location="cpu", weights_only=False)
                conv.load_state_dict(sd, strict=True)
            else:
                logger.info(f"Fitting {combo} with closed-form least squares")
                conv = fit_conv_svd(conv, src_train, tgt_train)
                torch.save(conv.state_dict(), state_path)
                logger.info(f"Fitted stitching layer saved at {state_path}")

            train_metrics = evaluate_stitching_layer_spatial(
                stitching_layer=conv, Z_batches=src_train, Y_batches=tgt_train,
                device=device,
            )
            logger.info(f"[{combo} TRAIN]  MSE={train_metrics['mse']:.6f}")

            val_metrics = evaluate_stitching_layer_spatial(
                stitching_layer=conv, Z_batches=src_val, Y_batches=tgt_val,
                device=device,
            )
            logger.info(f"[{combo} VAL]    MSE={val_metrics['mse']:.6f}")

            with open(
                os.path.join(args.feature_save_path, f"metrics_{combo}_post.txt"),
                "w",
            ) as f:
                f.write(f"=== TRAIN ({n_train} samples) ===\n")
                for k, v in train_metrics.items():
                    f.write(f"{k}: {v}\n")
                f.write(f"\n=== VAL ({n_val} samples) ===\n")
                for k, v in val_metrics.items():
                    f.write(f"{k}: {v}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = find_layer_stitching_argument()
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.5,
        help="Fraction of samples to hold out for validation (default: 0.5).",
    )

    args = parser.parse_args()
    main(args)
