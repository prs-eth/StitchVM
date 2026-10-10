import argparse
import json
import math
import os
from pathlib import Path

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, set_seed
from loguru import logger
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from config.stitching_config.base import stitching_training_argument
from stitchvm.utils.utils_for_thirdparty import (
    load_clip_model,
    load_flux_dit,
    load_flux_prompt_embedding,
    load_flux_vae,
    load_sd3_dit,
    load_sd3_prompt_embedding,
    load_sd3_vae,
)
from stitchvm.data.dataset_util import create_stitching_dataloader
from stitchvm.losses.aesthetic_loss import LossAestheticStitched
from stitchvm.losses.clip_loss import LossCLIPStitched
from stitchvm.models.backbones.flux import FluxDiTStitched
from stitchvm.models.backbones.sd3 import SD3DiTStitched
from stitchvm.models.feedforward.aesthetic_head import AestheticEncoderHead
from stitchvm.models.feedforward.clip_head import CLIPEncoderHead
from stitchvm.models.feedforward.dfn_clip_head import (
    DFNClipEncoderHead,
    DFNClipTeacherWrapper,
)
from stitchvm.models.feedforward.hpsv2_head import HPSv2EncoderHead, HPSv2TeacherWrapper
from stitchvm.models.stitched_image_model import StitchImage2D
from stitchvm.utils.ema import EMAState
from stitchvm.utils.noisy_latent import make_noisy_latent


def _checkpoint_dir(save_path: str, epoch_next: int) -> Path:
    return Path(save_path) / "checkpoints" / f"epoch_{epoch_next:04d}"


def _atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _gather_model_trainables(stitched_model, accelerator):
    """Inference-relevant weights: stitching_layer + full head (stitched_3d_model)."""
    m = accelerator.unwrap_model(stitched_model)
    return {
        "stitching_layer": {
            k: v.detach().cpu().clone() for k, v in m.stitching_layer.state_dict().items()
        },
        "stitched_3d_model": {
            k: v.detach().cpu().clone() for k, v in m.stitched_3d_model.state_dict().items()
        },
    }


def save_checkpoint_end_of_epoch(
    *,
    args,
    accelerator,
    stitched_model,
    epoch_next: int,
    ema_state: "EMAState | None" = None,
):
    """
    Saves one (or two) files per epoch:
      - model_trainables.pt          (raw weights)
      - model_trainables_ema.pt      (only when --ema_decay > 0 — EMA weights)
    """
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    ckpt_dir = _checkpoint_dir(args.save_path, epoch_next)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Stamp the architecture identity so eval can assert it matches.
    location = args.stitching_layer_location

    model_state = _gather_model_trainables(stitched_model, accelerator)
    model_state["stitching_layer_location"] = location
    _atomic_torch_save(model_state, ckpt_dir / "model_trainables.pt")

    if ema_state is not None:
        m = accelerator.unwrap_model(stitched_model)
        backup = ema_state.apply_to(m)
        try:
            ema_model_state = _gather_model_trainables(stitched_model, accelerator)
            ema_model_state["stitching_layer_location"] = location
        finally:
            ema_state.restore(m, backup)
        _atomic_torch_save(ema_model_state, ckpt_dir / "model_trainables_ema.pt")

    accelerator.wait_for_everyone()


def training_one_epoch_loop_clip(
    stitched_model,
    teacher_model,
    optimizer,
    scheduler,
    train_dataloader,
    loss_criterion,
    epoch,
    accelerator,
    args,
    *,
    clip_processor,
    prompt_embedding,
    ema_state: "EMAState | None" = None,
):
    """One training epoch of Stage-2 distillation.

    Args:
        stitched_model: DDP-wrapped StitchImage2D
        teacher_model: frozen reward encoder (CLIPVisionModelWithProjection, or
            the HPSv2/DFN teacher wrappers — anything exposing .image_embeds and
            .last_hidden_state on its output), not DDP-wrapped
        clip_processor: processor providing image_mean/std for the teacher
        prompt_embedding: (prompt_embeds, pooled_prompt_embeds) tuple for the DiT
    """
    clip_normalize = torchvision.transforms.Normalize(
        mean=clip_processor.image_mean,
        std=clip_processor.image_std,
    )
    prompt_embeds, pooled_embeds = prompt_embedding

    # Metrics always go to the terminal (so a run is inspectable without a
    # wandb account). The JSONL file is written only when $LOG_DIR is set;
    # main() resolved and created it at launch (args.log_path).
    log_path = getattr(args, "log_path", None)
    opt_step = 0

    for step, data in enumerate(tqdm(train_dataloader)):
        # `accumulate` marks every k-th micro-batch (and the dataloader's last
        # one) as a sync step: accelerator.backward scales the loss by 1/k,
        # the prepared optimizer's step()/zero_grad() no-op on non-sync
        # micro-batches, and DDP skips gradient all-reduce until sync. The
        # clip / scheduler / EMA boundary ops below are gated explicitly.
        with accelerator.accumulate(stitched_model):
            ff_images = data["feedforward_image_tensor"].to(accelerator.device)

            # Encode in fp32 before adding noise for the DiT input.
            m = accelerator.unwrap_model(stitched_model)
            with torch.no_grad():
                latent = m.vae_encoder_forward(
                    data["vae_image_tensor"].to(accelerator.device)
                )
                noisy_latent, sigma = make_noisy_latent(
                    latent,
                    args.timestep_range,
                    distribution=args.timestep_distribution,
                )

            ff_images_clip = clip_normalize((ff_images + 1) / 2)

            with accelerator.autocast():
                # Student forward
                B = noisy_latent.shape[0]
                student_output = stitched_model(
                    noisy_latent=noisy_latent,
                    timestep=sigma * 1000,
                    encoder_hidden_states=prompt_embeds.expand(B, -1, -1),
                    pooled_projections=pooled_embeds.expand(B, -1),
                    train=True,
                )

                # Teacher forward on the reward-model-resolution images
                with torch.no_grad():
                    teacher_output = teacher_model(pixel_values=ff_images_clip)

                loss = loss_criterion(student_output, teacher_output)
                total_loss = loss["total_loss"]

            invalid_loss = (~torch.isfinite(total_loss.detach())).to(dtype=torch.int32)
            if accelerator.reduce(invalid_loss, reduction="sum").item():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {step}; "
                    "stopped before backward and the optimizer update."
                )
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                all_trainable = [
                    p for p in stitched_model.parameters() if p.requires_grad
                ]
                gradnorm = accelerator.clip_grad_norm_(all_trainable, max_norm=1.0)
                invalid = (~torch.isfinite(gradnorm)).to(dtype=torch.int32)
                invalid = accelerator.reduce(invalid, reduction="sum")
                if invalid.item():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch {epoch}, step {step}; "
                        "stopped before the optimizer update."
                    )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                scheduler.step()
                opt_step += 1
                if ema_state is not None:
                    ema_state.update(accelerator.unwrap_model(stitched_model))

            if accelerator.is_main_process and accelerator.sync_gradients:
                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/token_loss": loss["token_loss"].item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/gradnorm": gradnorm,
                    "train/sigma_mean": sigma.mean().item(),
                }
                if "embedding_loss" in loss:
                    log_dict["train/embedding_loss"] = loss["embedding_loss"].item()
                if "score_loss" in loss:
                    log_dict["train/score_loss"] = loss["score_loss"].item()
                    log_dict["train/student_aesthetic_score"] = loss[
                        "student_aesthetic_score"
                    ].item()
                    log_dict["train/teacher_aesthetic_score"] = loss[
                        "teacher_aesthetic_score"
                    ].item()
                accelerator.log(log_dict)

                # Terminal + file, independent of wandb. gradnorm arrives as a
                # tensor, so flatten to plain numbers before formatting/JSON.
                flat = {
                    k: (v.item() if torch.is_tensor(v) else v)
                    for k, v in log_dict.items()
                }
                # A NaN/Inf is reported on the step it appears, never swallowed
                # by the --log_every interval.
                nonfinite = [
                    k.split("/", 1)[-1] for k, v in flat.items()
                    if isinstance(v, float) and not math.isfinite(v)
                ]
                if opt_step % max(1, args.log_every) == 0 or nonfinite:
                    parts = " ".join(
                        f"{k.split('/', 1)[-1]}="
                        + (f"{v:.4e}" if isinstance(v, float) else f"{v}")
                        for k, v in flat.items()
                        if k not in ("train/epoch", "train/step")
                    )
                    tag = ("  !! NON-FINITE: " + ",".join(nonfinite)) if nonfinite else ""
                    logger.info(f"[e{epoch} s{step} opt{opt_step}] {parts}{tag}")
                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"opt_step": opt_step, **flat}) + "\n")


def _setup_full_finetune(stitched_model, args):
    """Full fine-tuning of the feedforward head (stitched_3d_model)."""
    for param in stitched_model.stitched_3d_model.parameters():
        param.requires_grad = True
    # Aesthetic MLP stays frozen — only the encoder suffix + stitching layer train.
    if args.feedforward_model == "aesthetic":
        for param in stitched_model.stitched_3d_model.aesthetic_mlp.parameters():
            param.requires_grad = False


def _get_image_diffusion_loaders(args):
    """Pick the (vae_loader, dit_loader, text_loader, DiTStitchedCls) quadruple
    based on ``args.diffusion_model``. Centralizes the SD3 vs Flux dispatch
    for ``_build_image_stitched_model`` and ``_build_hpsv2_stitched_model``
    so adding more image backbones later only touches one place.
    """
    if args.diffusion_model == "flux_dev":
        return (
            load_flux_vae,
            load_flux_dit,
            load_flux_prompt_embedding,
            FluxDiTStitched,
        )
    # Default = SD3 path (sd3.5 medium or sd3.5_large)
    return (
        load_sd3_vae,
        load_sd3_dit,
        load_sd3_prompt_embedding,
        SD3DiTStitched,
    )


def _build_dit_stitched(args, dit_full, dit_split, DiTStitchedCls):
    """Construct the DiT stitching wrapper, forwarding Flux-specific kwargs
    only when the class actually accepts them. Keeps the call sites in
    ``_build_image_stitched_model`` / ``_build_hpsv2_stitched_model`` tidy.
    """
    if DiTStitchedCls is FluxDiTStitched:
        return FluxDiTStitched(
            dit_full,
            dit_split,
            guidance_scale=args.flux_guidance_scale,
        )
    return DiTStitchedCls(dit_full, dit_split)


def _split_at_stitching_layer(location: str) -> tuple[int, int]:
    """Parse 'enc_blocks_N_dit_M' -> (enc_split=N, dit_split=M).

    Every release stitching point has this exact form; anything else is
    rejected loudly (e.g. a bare 'dit_13' would otherwise misparse).
    """
    import re

    m = re.fullmatch(r"enc_blocks_(\d+)_dit_(\d+)", location)
    if not m:
        raise ValueError(
            f"Invalid --stitching_layer_location {location!r}: expected "
            "'enc_blocks_N_dit_M' (e.g. enc_blocks_1_dit_4)."
        )
    return int(m.group(1)), int(m.group(2))


def _assemble_stitched(args, device, clip_head, head_label):
    """Common assembly: VAE + DiT prefix + StitchImage2D + freeze policy.

    Returns the assembled StitchImage2D and the (prompt_embeds, pooled) tuple.
    """
    vae_loader, dit_loader, text_loader, DiTStitchedCls = _get_image_diffusion_loaders(args)
    backbone_tag = args.diffusion_model

    vae = vae_loader(args, device)
    vae.eval()

    _, dit_split = _split_at_stitching_layer(args.stitching_layer_location)
    logger.info(f"Stitching with {backbone_tag} DiT and {head_label} head.")
    dit_full = dit_loader(args, device)
    dit_model = _build_dit_stitched(args, dit_full, dit_split, DiTStitchedCls)
    prompt_embedding = text_loader(args, device)

    stitched_model = StitchImage2D(
        diffusion_vae=vae,
        clip_head=clip_head,
        stitching_layer_config=args.stitching_layer_config,
        stitching_layer_init_path=args.initialization_weight_path,
        device=device,
        dit=dit_model,
    )

    # Freeze VAE + DiT; train stitching layer (head trainability set by _setup_full_finetune)
    for param in stitched_model.diffusion_vae.parameters():
        param.requires_grad = False
    for param in stitched_model.stitching_layer.parameters():
        param.requires_grad = True
    for param in stitched_model.stitched_dit.parameters():
        param.requires_grad = False

    return stitched_model, prompt_embedding


def _build_image_stitched_model(args, device):
    """CLIP or Aesthetic head + SD3/Flux backbone."""
    if args.feedforward_model == "aesthetic":
        args.clip_model_id = "openai/clip-vit-large-patch14"
        logger.info("Aesthetic mode: using openai/clip-vit-large-patch14")
        assert args.aesthetic_mlp_path is not None, (
            "--aesthetic_mlp_path is required when --feedforward_model=aesthetic"
        )

    feed_forward_model, clip_processor = load_clip_model(args, device)
    feed_forward_model.eval()

    clip_split, _ = _split_at_stitching_layer(args.stitching_layer_location)
    if args.feedforward_model == "aesthetic":
        clip_head = AestheticEncoderHead.from_clip_vision_model(
            feed_forward_model,
            clip_split,
            mlp_weights_path=args.aesthetic_mlp_path,
        )
    else:
        clip_head = CLIPEncoderHead.from_clip_vision_model(feed_forward_model, clip_split)

    stitched_model, prompt_embedding = _assemble_stitched(args, device, clip_head, "CLIP")
    return stitched_model, feed_forward_model, clip_processor, prompt_embedding


def _build_hpsv2_stitched_model(args, device):
    """HPSv2 head + SD3/Flux backbone."""
    from stitchvm.utils.utils_for_thirdparty import load_hpsv2_model

    hpsv2_model, hpsv2_processor = load_hpsv2_model(args, device)

    hpsv2_split, _ = _split_at_stitching_layer(args.stitching_layer_location)
    hpsv2_head = HPSv2EncoderHead.from_hpsv2_model(hpsv2_model, hpsv2_split)

    stitched_model, prompt_embedding = _assemble_stitched(args, device, hpsv2_head, "HPSv2")
    teacher = HPSv2TeacherWrapper(hpsv2_model)
    teacher.eval()
    return stitched_model, teacher, hpsv2_processor, prompt_embedding


def _build_dfn_clip_stitched_model(args, device):
    """DFN5B-CLIP head + SD3/Flux backbone (open_clip ViT-H-14)."""
    from stitchvm.utils.utils_for_thirdparty import load_dfn_clip_model

    dfn_model, dfn_processor = load_dfn_clip_model(args, device)

    dfn_split, _ = _split_at_stitching_layer(args.stitching_layer_location)
    dfn_head = DFNClipEncoderHead.from_dfn_clip_model(dfn_model, dfn_split)

    stitched_model, prompt_embedding = _assemble_stitched(args, device, dfn_head, "DFN CLIP")
    teacher = DFNClipTeacherWrapper(dfn_model)
    teacher.eval()
    return stitched_model, teacher, dfn_processor, prompt_embedding


def main(args: argparse.Namespace):
    set_seed(23456)
    if Path(args.save_path).exists():
        raise FileExistsError(f"Use a new --save_path: {args.save_path}")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema_decay must be in [0, 1)")

    # ---------------------------------------------------------------
    # Speed optimizations (gated by --optimize_speed)
    # ---------------------------------------------------------------
    # Default policy:
    #   --optimize_speed unset  →  ON for flux_dev, OFF for every other backbone
    #   --optimize_speed / --no-optimize_speed  →  explicit override
    # When ON, we enable: TF32 matmul/cudnn (a reduced-precision fp32 matmul
    # path on Ampere+), cudnn.benchmark, fused AdamW (later in this function),
    # and clip_head gradient_checkpointing=False (also later). The release
    # runs pass --optimize_speed explicitly for every backbone.
    if getattr(args, "optimize_speed", None) is None:
        args.optimize_speed = args.diffusion_model == "flux_dev"

    if args.optimize_speed:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        logger.info(
            "optimize_speed=True: TF32 matmul/cudnn enabled, cudnn.benchmark=True, "
            "float32_matmul_precision=high"
        )
    else:
        logger.info(
            "optimize_speed=False: using PyTorch defaults "
            "(no TF32, no cudnn.benchmark)"
        )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"--gradient_accumulation_steps must be >= 1, "
            f"got {args.gradient_accumulation_steps}"
        )

    loggers = ["wandb"] if args.wandb_logging else None

    # Passing an explicit plugin makes the CLI flag authoritative — a
    # gradient_accumulation_steps entry in a saved `accelerate config` can
    # never silently override it.
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with=loggers,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps
        ),
    )
    global_batch = (
        accelerator.num_processes * args.batch_size * args.gradient_accumulation_steps
    )
    logger.info(
        f"Nominal global batch: {global_batch} = "
        f"{accelerator.num_processes} processes × {args.batch_size} images/device × "
        f"{args.gradient_accumulation_steps} accumulation steps "
        f"(the final update of an epoch can contain fewer micro-batches)"
    )
    if args.wandb_logging and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project_name,
            config=args,
            init_kwargs={"wandb": {"name": args.save_path}},
        )

    # Resolve the metrics file up front, so the directory exists (and its
    # location is reported) immediately at launch rather than minutes later
    # when the first epoch starts.
    log_dir = os.environ.get("LOG_DIR")
    args.log_path = (
        str(Path(log_dir).resolve() / f"{Path(args.save_path).name}.jsonl")
        if log_dir else None
    )
    if accelerator.is_main_process:
        if args.log_path:
            Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Per-step metrics -> {args.log_path}")
        else:
            logger.info(
                "LOG_DIR not set: metrics go to the terminal only "
                f"(every --log_every={args.log_every} optimizer steps). "
                "Set LOG_DIR=<dir> to also write a JSONL record."
            )

    device = accelerator.device

    # ---------------------------------------------------------------
    # 1. Build stitched model + teacher
    # ---------------------------------------------------------------
    logger.info("Initializing stitching model")
    if args.feedforward_model == "hpsv2":
        stitched_model, feed_forward_model, clip_processor, prompt_embedding = (
            _build_hpsv2_stitched_model(args, device)
        )
        loss_criterion = LossCLIPStitched(
            loss_type=args.loss_type,
            token_weight=args.token_weight,
        )
    elif args.feedforward_model == "dfn_clip":
        stitched_model, feed_forward_model, clip_processor, prompt_embedding = (
            _build_dfn_clip_stitched_model(args, device)
        )
        loss_criterion = LossCLIPStitched(
            loss_type=args.loss_type,
            token_weight=args.token_weight,
        )
    elif args.feedforward_model == "aesthetic":
        stitched_model, feed_forward_model, clip_processor, prompt_embedding = (
            _build_image_stitched_model(args, device)
        )
        aesthetic_mlp = stitched_model.stitched_3d_model.aesthetic_mlp
        aesthetic_mlp.eval()
        loss_criterion = LossAestheticStitched(
            aesthetic_mlp,
            token_weight=args.token_weight,
            embed_weight=args.embed_weight,
            score_weight=args.score_weight,
            loss_type=args.loss_type,
        )
    else:
        # clip (default)
        stitched_model, feed_forward_model, clip_processor, prompt_embedding = (
            _build_image_stitched_model(args, device)
        )
        loss_criterion = LossCLIPStitched(
            loss_type=args.loss_type,
            token_weight=args.token_weight,
        )

    # ---------------------------------------------------------------
    # 2. Full fine-tuning setup
    # ---------------------------------------------------------------
    _setup_full_finetune(stitched_model, args)
    total_params = sum(p.numel() for p in stitched_model.parameters())
    trainable_params = sum(
        p.numel() for p in stitched_model.parameters() if p.requires_grad
    )
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ---------------------------------------------------------------
    # 3. Freeze teacher; configure gradient checkpointing on the head
    # ---------------------------------------------------------------
    for param in feed_forward_model.parameters():
        param.requires_grad = False

    stitched_model.to(device=device)
    # Checkpoint head activations when memory saving is requested.
    stitched_model.stitched_3d_model.gradient_checkpointing = not args.optimize_speed
    logger.info(
        f"stitched_3d_model.gradient_checkpointing = "
        f"{stitched_model.stitched_3d_model.gradient_checkpointing} "
        f"(optimize_speed={args.optimize_speed})"
    )

    # ---------------------------------------------------------------
    # 4. Dataloader, optimizer, scheduler
    # ---------------------------------------------------------------
    logger.info("Creating dataloader")
    train_dataloader, _distributed_sampler = create_stitching_dataloader(
        args.dataset, args, augmentation=True
    )

    logger.info("Setting up optimizer.")
    logger.info("configuration:Optimizer: AdamW")
    logger.info(f"configuration:Learning rate: {args.learning_rate}")
    logger.info(f"configuration:Weight decay: {args.weight_decay}")
    logger.info(f"configuration:Warmup steps: {args.warmup_steps}")

    stitch_lr_mult = args.stitching_lr_multiplier
    if stitch_lr_mult != 1.0:
        m_unwrap = accelerator.unwrap_model(stitched_model)
        stitch_ids = {id(p) for p in m_unwrap.stitching_layer.parameters()}
        stitch_params = [p for p in stitched_model.parameters()
                         if p.requires_grad and id(p) in stitch_ids]
        other_params = [p for p in stitched_model.parameters()
                        if p.requires_grad and id(p) not in stitch_ids]
        param_groups = [
            {"params": other_params, "lr": args.learning_rate},
            {"params": stitch_params, "lr": args.learning_rate * stitch_lr_mult},
        ]
        logger.info(
            f"Stitching layer LR: {args.learning_rate * stitch_lr_mult:.2e} "
            f"({stitch_lr_mult}x), other: {args.learning_rate:.2e}"
        )
    else:
        param_groups = [p for p in stitched_model.parameters() if p.requires_grad]
    # Fused AdamW is enabled by --optimize_speed.
    _optimizer_kwargs = dict(lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimize_speed:
        _optimizer_kwargs["fused"] = True
        logger.info("optimize_speed=True: using fused AdamW")
    optimizer = torch.optim.AdamW(param_groups, **_optimizer_kwargs)

    train_dataloader, optimizer, stitched_model = accelerator.prepare(
        train_dataloader, optimizer, stitched_model
    )

    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
    )

    # ---------------------------------------------------------------
    # 4b. EMA of the trainable parameters
    # ---------------------------------------------------------------
    ema_state = None
    if args.ema_decay > 0.0:
        ema_state = EMAState(
            accelerator.unwrap_model(stitched_model),
            decay=args.ema_decay,
        )
        n_tracked = sum(s.numel() for s in ema_state.shadow.values())
        logger.info(
            f"EMA enabled: decay={args.ema_decay}, "
            f"tracking {n_tracked:,} trainable params across "
            f"{len(ema_state.shadow)} tensors"
        )

    # ---------------------------------------------------------------
    # 5. Training loop
    # ---------------------------------------------------------------
    for epoch in range(args.num_epochs):
        stitched_model.eval()  # intended: no Dropout/BatchNorm among trainables
        feed_forward_model.eval()

        if hasattr(train_dataloader, "set_epoch"):
            logger.info("Setting epoch for distributed sampler")
            train_dataloader.set_epoch(epoch)
        # Accelerate's distributed wrapper does not forward set_epoch to the
        # nested custom batch sampler, so reseed that sampler explicitly.
        if hasattr(_distributed_sampler, "set_epoch"):
            _distributed_sampler.set_epoch(epoch)

        training_one_epoch_loop_clip(
            stitched_model,
            feed_forward_model,
            optimizer,
            scheduler,
            train_dataloader,
            loss_criterion,
            epoch,
            accelerator,
            args,
            clip_processor=clip_processor,
            prompt_embedding=prompt_embedding,
            ema_state=ema_state,
        )

        save_checkpoint_end_of_epoch(
            args=args,
            accelerator=accelerator,
            stitched_model=stitched_model,
            epoch_next=epoch + 1,
            ema_state=ema_state,
        )


if __name__ == "__main__":
    args = stitching_training_argument().parse_args()
    main(args)
