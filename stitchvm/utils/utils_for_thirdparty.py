import argparse
import os

import torch
from loguru import logger


_SD3_MODEL_IDS = {
    "sd3.5": "stabilityai/stable-diffusion-3.5-medium",
    "sd3.5_large": "stabilityai/stable-diffusion-3.5-large",
}

_FLUX_MODEL_IDS = {
    "flux_dev": "black-forest-labs/FLUX.1-dev",
}


def _get_sd3_model_id(args):
    dm = getattr(args, "diffusion_model", None)
    if dm in _SD3_MODEL_IDS:
        return _SD3_MODEL_IDS[dm]
    raise NotImplementedError(f"Image diffusion model {dm} is not implemented.")


def _get_flux_model_id(args):
    dm = getattr(args, "diffusion_model", None)
    if dm in _FLUX_MODEL_IDS:
        return _FLUX_MODEL_IDS[dm]
    raise NotImplementedError(f"Flux diffusion model {dm} is not implemented.")


def _sd3_revision_kwargs(args) -> dict:
    """``revision=`` for the SD3 loaders, only when ``args.diffusion_revision`` is set (the extension
    inference builders pin it); otherwise no extra argument, exactly as before."""
    revision = getattr(args, "diffusion_revision", None)
    return {"revision": revision} if revision else {}


def _maybe_slice_vae(vae) -> None:
    """Encode samples individually to reduce memory when STITCHVM_VAE_SLICING=1."""
    if os.environ.get("STITCHVM_VAE_SLICING", "0").strip().lower() in ("1", "true", "yes", "on"):
        vae.enable_slicing()
        logger.info("STITCHVM_VAE_SLICING=1: VAE encode sliced (batch-independent peak)")


def load_sd3_dit(args: argparse.Namespace, device: torch.device):
    """Load SD3.5 DiT transformer."""
    from diffusers import SD3Transformer2DModel

    model_id = _get_sd3_model_id(args)

    logger.info(f"Loading SD3 DiT from {model_id} with device {device}")
    dit = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=_hf_offline(), **_sd3_revision_kwargs(args),
    )
    dit.to(device)
    dit.eval()
    return dit


def load_sd3_prompt_embedding(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute empty-prompt embeddings for SD3.5.

    Loads the pipeline with transformer=None and vae=None to avoid
    loading heavyweight components, computes text embeddings, then frees.

    Returns:
        (prompt_embeds, pooled_prompt_embeds)
    """
    from diffusers import StableDiffusion3Pipeline

    model_id = _get_sd3_model_id(args)

    logger.info(f"Loading SD3 text encoders from {model_id} for prompt embedding")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, transformer=None, vae=None, torch_dtype=torch.bfloat16,
        local_files_only=_hf_offline(), **_sd3_revision_kwargs(args),
    )
    # Move only text encoders to device
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)
    if pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.to(device)

    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            device=device,
        )

    del pipe
    torch.cuda.empty_cache()
    logger.info(
        f"SD3 prompt embeddings: prompt={prompt_embeds.shape}, "
        f"pooled={pooled_prompt_embeds.shape}"
    )
    return prompt_embeds, pooled_prompt_embeds


def load_sd3_vae(args: argparse.Namespace, device: torch.device):
    """Load SD3.5 VAE based on the provided arguments."""
    from diffusers import AutoencoderKL

    model_id = _get_sd3_model_id(args)

    logger.info(f"Loading SD3 VAE from {model_id} with device {device}")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
        local_files_only=_hf_offline(), **_sd3_revision_kwargs(args),
    )
    vae.to(device)
    _maybe_slice_vae(vae)
    return vae


def _hf_offline() -> bool:
    """Return True when the HF hub is in offline mode. Used on offline compute
    nodes to force diffusers into local-only mode so sharded checkpoints load
    without calling ``model_info()``.

    Mirrors huggingface_hub's own truthy set (1/true/yes/on, case-insensitive)
    and also honors TRANSFORMERS_OFFLINE, so the loaders can never disagree
    with the hub about being offline.
    """
    truthy = ("1", "true", "yes", "on")
    return any(
        os.environ.get(var, "").strip().lower() in truthy
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def load_flux_vae(args: argparse.Namespace, device: torch.device):
    """Load Flux 1.0 [dev] VAE (standard AutoencoderKL, 16 latent channels)."""
    from diffusers import AutoencoderKL

    model_id = _get_flux_model_id(args)

    logger.info(f"Loading Flux VAE from {model_id} with device {device}")
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=_hf_offline(),
    )
    vae.to(device)
    _maybe_slice_vae(vae)
    return vae


def load_flux_dit(args: argparse.Namespace, device: torch.device):
    """Load Flux 1.0 [dev] transformer (FluxTransformer2DModel).

    Flux-dev's transformer is sharded across 3 safetensors files. On
    ``from_pretrained`` diffusers calls ``model_info()`` to enumerate the
    shards, which requires network access. When ``HF_HUB_OFFLINE=1`` is
    set we must pass ``local_files_only=True``
    explicitly to skip that call and read only from the local HF cache.
    """
    from diffusers import FluxTransformer2DModel

    model_id = _get_flux_model_id(args)

    logger.info(f"Loading Flux DiT from {model_id} with device {device}")
    dit = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=_hf_offline(),
    )
    dit.to(device)
    dit.eval()
    return dit


def load_flux_prompt_embedding(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute empty-prompt embeddings for Flux 1.0 [dev].

    Loads ``FluxPipeline`` with ``transformer=None`` and ``vae=None`` to
    avoid pulling in the 12 GB transformer and the VAE — we only need
    the CLIP + T5 text encoders. Returns a ``(prompt_embeds, pooled_prompt_embeds)``
    tuple matching ``load_sd3_prompt_embedding``'s signature. The Flux
    pipeline also returns ``text_ids`` (zeros of shape ``(512, 3)``) but
    we drop it because :class:`FluxDiTStitched` recomputes it internally.

    Returns:
        ``(prompt_embeds, pooled_prompt_embeds)`` with shapes
        ``(1, 512, 4096)`` (T5) and ``(1, 768)`` (CLIP pooled).
    """
    from diffusers import FluxPipeline

    model_id = _get_flux_model_id(args)

    logger.info(f"Loading Flux text encoders from {model_id} for prompt embedding")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        local_files_only=_hf_offline(),
    )
    # Move only text encoders to device
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt="",
            prompt_2="",
            device=device,
            max_sequence_length=512,
            num_images_per_prompt=1,
        )

    del pipe
    torch.cuda.empty_cache()
    logger.info(
        f"Flux prompt embeddings: prompt={prompt_embeds.shape}, "
        f"pooled={pooled_prompt_embeds.shape}"
    )
    return prompt_embeds, pooled_prompt_embeds


def load_clip_model(args: argparse.Namespace, device: torch.device):
    """Load CLIP vision model (with projection) and its image processor.

    Returns:
        (CLIPVisionModelWithProjection, CLIPImageProcessor) tuple.
    """
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    model_id = getattr(args, "clip_model_id", "openai/clip-vit-large-patch14-336")
    logger.info(f"Loading CLIP vision model from {model_id} with device {device}")
    # Optionally select the CLIP attention implementation, e.g. "sdpa" or
    # "eager", for compatibility with the installed attention libraries.
    _extra = {}
    _attn_impl = os.environ.get("STITCHVM_ATTN_IMPL")
    if _attn_impl:
        _extra["attn_implementation"] = _attn_impl
        logger.info(f"STITCHVM_ATTN_IMPL={_attn_impl}: forcing CLIP attention impl")
    clip = CLIPVisionModelWithProjection.from_pretrained(
        model_id, torch_dtype=torch.float32, local_files_only=_hf_offline(), **_extra
    )
    logger.info(f"CLIP attention implementation: {clip.config._attn_implementation}")
    clip.to(device)
    clip.eval()
    processor = CLIPImageProcessor.from_pretrained(
        model_id, local_files_only=_hf_offline()
    )
    return clip, processor


def load_hpsv2_model(args: argparse.Namespace, device: torch.device):
    """Load HPSv2 model (open_clip ViT-H-14 + HPSv2 fine-tuned weights).

    Requires args.hpsv2_checkpoint_dir containing:
      - open_clip_pytorch_model.bin (base ViT-H-14 weights)
      - HPS_v2.1_compressed.pt (HPSv2 fine-tuned weights)

    Returns:
        (hpsv2_model, HPSv2Processor) tuple
    """
    from hpsv2.src.open_clip import create_model

    from stitchvm.models.feedforward.hpsv2_head import HPSv2Processor

    ckpt_dir = args.hpsv2_checkpoint_dir
    assert ckpt_dir is not None, (
        "--hpsv2_checkpoint_dir is required when --feedforward_model=hpsv2"
    )
    base_weights = os.path.join(ckpt_dir, "open_clip_pytorch_model.bin")
    hpsv2_weights = os.path.join(ckpt_dir, "HPS_v2.1_compressed.pt")

    logger.info(f"Loading HPSv2 model from {ckpt_dir}")
    model = create_model(
        "ViT-H-14",
        base_weights,
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        output_dict=True,
    )

    checkpoint = torch.load(hpsv2_weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    logger.info("HPSv2 model loaded (ViT-H-14, 32 blocks, hidden=1280, proj=1024)")
    return model, HPSv2Processor()


def load_dfn_clip_model(args: argparse.Namespace, device: torch.device):
    """Load Apple DFN5B CLIP (open_clip ViT-H-14 from `hf-hub:apple/DFN5B-CLIP-ViT-H-14-378`).

    Returns:
        (open_clip model, DFNClipProcessor) tuple. The processor exposes
        ``image_mean`` / ``image_std`` so the training/eval normalize transforms
        line up with what the model was trained on (OpenAI CLIP stats).
    """
    import open_clip

    from stitchvm.models.feedforward.dfn_clip_head import DFNClipProcessor

    model_id = getattr(
        args, "dfn_clip_model_id", "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
    )
    logger.info(f"Loading DFN CLIP model {model_id} (open_clip)")
    model, _, _ = open_clip.create_model_and_transforms(model_id)
    model.to(device)
    model.eval()
    logger.info(
        "DFN CLIP loaded (ViT-H-14, 32 blocks, hidden=1280, proj=1024)"
    )
    return model, DFNClipProcessor()


def load_hpsv3_visual(args: argparse.Namespace, device: torch.device):
    """Load only the Qwen2-VL vision tower of HPSv3.

    Search and distillation never run the language branch, so only the
    `visual.*` weights of the reward checkpoint are materialized, on top of
    the vision config of the Qwen2-VL-7B-Instruct base model. The full reward
    model, language branch included, is loaded by
    stitchvm.models.qwen2vl_reward for evaluation.

    Returns:
        (Qwen2VisionTransformerPretrainedModel in bf16, Qwen2VLProcessorStats)
    """
    import safetensors
    from transformers import AutoConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel,
    )

    from stitchvm.models.feedforward.qwen2vl_vision_head import Qwen2VLProcessorStats
    from stitchvm.models import qwen2vl_reward as R

    path = getattr(args, "hpsv3_checkpoint", None) or R.hpsv3_checkpoint_file()
    sd = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("visual."):
                sd[k[len("visual."):]] = f.get_tensor(k)

    config = AutoConfig.from_pretrained(
        R.HPSV3_BASE_MODEL, revision=R.HPSV3_BASE_REVISION, local_files_only=_hf_offline()
    )
    vcfg = config.vision_config
    vcfg._attn_implementation = "sdpa"
    visual = Qwen2VisionTransformerPretrainedModel._from_config(vcfg, torch_dtype=torch.bfloat16)
    visual.load_state_dict(sd, strict=True)
    visual.to(device)
    visual.eval()
    logger.info(f"Loaded HPSv3 vision tower ({len(sd)} tensors) from {path}")
    return visual, Qwen2VLProcessorStats()
