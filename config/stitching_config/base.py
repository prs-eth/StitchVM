import argparse
import pathlib
import textwrap

from stitchvm.models.stitching_layer_builder import parse_conv_spec


def _dedent(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


def make_parser(
    description: str,
    *,
    formatter: type[argparse.HelpFormatter] = argparse.ArgumentDefaultsHelpFormatter,
) -> argparse.ArgumentParser:
    """
    Create a parser with a consistent formatter that shows defaults automatically.
    """
    return argparse.ArgumentParser(
        description=description,
        formatter_class=formatter,
    )


def parse_dataset(arg: str) -> tuple[str, pathlib.Path]:
    """
    Convert a CLI token of the form NAME:ROOT into (name, root)
    and validate a few basic things.
    """
    try:
        name, root = arg.split(":", 1)
    except ValueError:  # no “:”
        raise argparse.ArgumentTypeError(
            "Dataset must be NAME:PATH, e.g. HPDv2:/data/hpdv2"
        )

    root_path = pathlib.Path(root).expanduser()
    if not root_path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {root_path}")

    return name, root_path


def add_model_selection_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Model selection")
    g.add_argument(
        "--feedforward_model",
        type=str,
        default="clip",
        choices=["clip", "aesthetic", "hpsv2", "dfn_clip", "hpsv3", "qwen35"],
        help="Reward-model encoder used as the StitchVM tail. hpsv3 stitches the "
        "Qwen2-VL vision branch of HPSv3; qwen35 stitches the vision branch of a "
        "UnifiedReward-2.0 Qwen3.5 checkpoint (--qwen35_checkpoint). Their language "
        "branches are not loaded for search or distillation.",
    )
    g.add_argument(
        "--qwen35_checkpoint",
        type=str,
        default=None,
        help=(
            "Local directory of a UnifiedReward-2.0 Qwen3.5 checkpoint (config.json, "
            "processor_config.json, safetensors, PINNED_REVISION written by "
            "scripts/download_weights.py). Every vision and processor setting is read "
            "from it. Required when --feedforward_model=qwen35."
        ),
    )
    g.add_argument(
        "--diffusion_model",
        type=str,
        default="sd3.5",
        choices=["sd3.5", "sd3.5_large", "flux_dev"],
        help=(
            "Diffusion backbone used as the StitchVM head. "
            "Paper covers SD 3.5 Medium ('sd3.5'), SD 3.5 Large ('sd3.5_large'), "
            "and FLUX.1-dev ('flux_dev')."
        ),
    )
    g.add_argument(
        "--flux_guidance_scale",
        type=float,
        default=3.5,
        help=(
            "Guidance scalar fed to FluxDiTStitched's time embedding. "
            "Flux-dev is guidance-distilled and requires a scalar; "
            "3.5 matches the Flux pipeline default."
        ),
    )
    g.add_argument(
        "--clip_model_id",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="HuggingFace model ID for CLIP vision model",
    )
    g.add_argument(
        "--hpsv2_checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Directory containing HPSv2 checkpoints "
            "(open_clip_pytorch_model.bin + HPS_v2.1_compressed.pt). "
            "Required when --feedforward_model=hpsv2."
        ),
    )
    g.add_argument(
        "--hpsv3_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to HPSv3.safetensors. Default: MizzenAI/HPSv3 from the Hub at the "
            "pinned revision. Used when --feedforward_model=hpsv3."
        ),
    )
    g.add_argument(
        "--dfn_clip_model_id",
        type=str,
        default="hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
        help=(
            "open_clip identifier for the DFN CLIP teacher (378 = 27x14 patch grid). "
            "Used when --feedforward_model=dfn_clip."
        ),
    )
    return parser


def parse_float_tuple(s: str) -> tuple:
    """Parse 'low,high' into (low, high), validated as a sigma range in [0, 1]."""
    try:
        parts = [float(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Cannot parse '{s}' as comma-separated floats. Expected format: low,high"
        )
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected 2 values (low,high), got {len(parts)}: '{s}'"
        )
    if parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError(
            f"low must be < high, got {parts[0]} >= {parts[1]}"
        )
    if parts[0] < 0.0 or parts[1] > 1.0:
        raise argparse.ArgumentTypeError(
            f"sigma range must lie within [0, 1], got {parts}"
        )
    return tuple(parts)


def add_dit_stitching_search_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add arguments specific to the DiT stitching layer search script.
    Call this in find_layer_stitching_argument() in base.py.
    """
    g = parser.add_argument_group("DiT stitching search")

    g.add_argument(
        "--timestep_range",
        type=parse_float_tuple,
        default=(0.0, 1.0),
        metavar="LOW,HIGH",
        help=(
            "Sigma range for flow matching noise during DiT feature extraction. "
            "Sigma is sampled uniformly from [LOW, HIGH]. "
            "The paper searches over the full range 0.0,1.0 (default)."
        ),
    )
    return parser


def _nonneg_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def parse_resolution_bucket(spec: str) -> tuple[int, int]:
    """Parse a square size or WIDTHxHEIGHT; VAE/DiT require multiples of 16."""
    parts = spec.lower().split("x")
    try:
        if len(parts) == 1:
            width = height = int(parts[0])
        elif len(parts) == 2:
            width, height = map(int, parts)
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{spec}'; use N or WIDTHxHEIGHT."
        ) from None
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise argparse.ArgumentTypeError(
            f"Resolution {width}x{height} must have positive dimensions divisible by 16."
        )
    return width, height


def add_training_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Shared image-loading arguments for feature extraction and training.
    """

    g = parser.add_argument_group("Image data")
    g.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    g.add_argument(
        "--num_workers",
        type=_nonneg_int,
        default=3,
        help=(
            "DataLoader worker processes (>= 0). Image reads are "
            "latency-bound on slow storage; "
            "adjust this value to hide read latency. "
            "0 loads in the main process."
        ),
    )
    g.add_argument(
        "--prefetch_factor",
        type=_positive_int,
        default=2,
        help=(
            "Batches prefetched per worker (>= 1; only used when "
            "--num_workers > 0). Raise to ~4-6 on high-latency storage to "
            "absorb read stalls. RAM cost scales with "
            "num_workers x prefetch_factor x batch_size images."
        ),
    )
    g.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset,
        metavar="NAME:ROOT",
        help=_dedent(
            """
            Provide one or several datasets as NAME:ROOT. Supported names:
              HPDv2  (unlabeled images from HPDv2 train; ~430K JPEGs)
              ava    (AVA training split — first ~235K images)
            Example combining both (paper §5.1 protocol):
              --dataset HPDv2:/data/hpdv2 \
              --dataset ava:/data/ava
            """
        ),
    )
    g.add_argument(
        "--width", type=int, default=512, help="VAE input width"
    )
    g.add_argument(
        "--height", type=int, default=512, help="VAE input height"
    )
    g.add_argument(
        "--resolution_buckets",
        type=parse_resolution_bucket,
        nargs="+",
        default=None,
        metavar="WxH",
        help="VAE input resolutions, sampled uniformly per batch. Use N for a "
        "square or WIDTHxHEIGHT for a rectangle; dimensions must be divisible "
        "by 16. Overrides --width/--height. Reward inputs retain their native "
        "size. When omitted, uses fixed --width x --height.",
    )
    g.add_argument(
        "--exclude_image_list",
        type=str,
        default=None,
        help="Text file of training images to drop, one path per line, relative to the "
        "dataset root (e.g. train/00020324.jpg for HPDv2) or absolute. Used by the "
        "UnifiedReward-2.0 Qwen3.5 recipes to remove near-duplicates of VL-RewardBench "
        "test images. Default: none.",
    )

    return parser


def additional_stitching_dataset_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Stitching dataset additional args")
    g.add_argument(
        "--feedforward_width",
        type=int,
        required=True,
        help="Reward-model input width in pixels: 224 for HPSv2/Aesthetic, "
        "336 for CLIP-L/14-336, 378 for DFN-CLIP, 448 for HPSv3. qwen35 reads "
        "the VAE image at its processor size instead; its recipes pass 512.",
    )
    g.add_argument(
        "--feedforward_height",
        type=int,
        required=True,
        help="Reward-model input height in pixels (same values as "
        "--feedforward_width).",
    )
    return parser


def find_layer_stitching_argument() -> argparse.ArgumentParser:
    parser = make_parser("Stage-1 stitching-layer search")

    g = parser.add_argument_group("Feature extraction")
    g.add_argument(
        "--feature_save_path",
        type=str,
        required=True,
        help="Path to save features used for searching stitching layer",
    )
    g.add_argument(
        "--iterations_for_feature_extraction",
        type=int,
        default=100,
        help=(
            "Number of iterations for feature extraction. "
            "Total data used = batch_size × iterations_for_feature_extraction"
        ),
    )
    g.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default="conv2d_k1x1_o1024_s1x1_p0x0",
        metavar="CONV_SPEC",
        help="Candidate stitching-conv spec, e.g. conv2d_k1x1_o1280_s1x1_p0x0 "
        "(o must match the reward head's hidden size: 1024 for CLIP-L, "
        "1280 for HPSv2/DFN ViT-H and HPSv3; the vision hidden size for qwen35).",
    )

    add_model_selection_args(parser)
    add_training_data_args(parser)
    additional_stitching_dataset_args(parser)
    add_dit_stitching_search_args(parser)
    return parser


def add_stitching_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Stitching")
    g.add_argument(
        "--stitching_layer_location",
        type=str,
        required=True,
        help="Stitching point 'enc_blocks_N_dit_M': hand over from DiT block M "
        "after reward-encoder block N (see the README layer selections).",
    )
    g.add_argument(
        "--initialization_weight_path",
        type=str,
        default=None,
        help="Path to the initialization weight for the stitching layer",
    )
    g.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default="conv2d_k1x1_o1024_s1x1_p0x0",
        metavar="CONV_SPEC",
        help="See --help in training script for grammar.",
    )
    g.add_argument(
        "--stitching_lr_multiplier",
        type=float,
        default=1.0,
        help=(
            "Learning rate multiplier for the stitching layer parameters. "
            "Effective LR = learning_rate * stitching_lr_multiplier. "
        ),
    )
    g.add_argument(
        "--optimize_speed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable training speed optimizations (TF32 matmul/cudnn, cudnn.benchmark, "
            "high float32_matmul_precision, fused AdamW, and "
            "clip_head.gradient_checkpointing=False). "
            "Default is auto: True for --diffusion_model flux_dev, False otherwise. "
            "Use --optimize_speed / --no-optimize_speed to override explicitly."
        ),
    )
    return parser


def add_run_and_logging_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Run & logging")
    g.add_argument(
        "--global_seed",
        type=int,
        default=23,
        help="Seed for the resolution-bucket sampler RNG (model/data seeding "
        "is fixed internally by the training script).",
    )

    g.add_argument(
        "--log_every",
        type=int,
        default=50,
        help=(
            "Print training metrics to the terminal every N optimizer steps. "
            "Set LOG_DIR to also write JSONL metrics. Non-finite losses or "
            "gradients stop training immediately, regardless of this interval."
        ),
    )

    # Correct boolean handling:
    # - exposes: --wandb_logging / --no-wandb_logging
    g.add_argument(
        "--wandb_logging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable wandb logging",
    )
    g.add_argument(
        "--wandb_project_name",
        type=str,
        default="stitchvm",
        help="Wandb project name",
    )
    return parser


def add_optimizer_args(
    parser: argparse.ArgumentParser,
    *,
    include_warmup: bool = True,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Optimizer")
    g.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    g.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    g.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients over this many micro-batches per optimizer "
            "step. Nominal global batch = total processes × batch_size × this. "
            "The Stage 2 launcher derives accumulation from NPROC, "
            "MICRO_BATCH_SIZE, and a target global batch of 256 for SD3.5 "
            "HPSv2/CLIP/Aesthetic or 128 for FLUX and all DFN. Direct Python "
            "invocations use this CLI value without automatic recipe selection."
        ),
    )

    if include_warmup:
        g.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")

    return parser


def add_training_loop_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Training loop")
    g.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    g.add_argument(
        "--save_path",
        type=str,
        default="outputs/stitchvm",
        help="Path to save checkpoints",
    )
    return parser


def stitching_training_argument() -> argparse.ArgumentParser:
    parser = make_parser("Stitching layer training arguments")
    add_model_selection_args(parser)
    parser.add_argument("--aesthetic_mlp_path", type=str, default=None,
                        help="Path to the frozen aesthetic scoring MLP checkpoint.")
    add_training_loop_args(parser)
    add_training_data_args(parser)
    add_stitching_args(parser)
    additional_stitching_dataset_args(parser)
    add_run_and_logging_args(parser)
    add_optimizer_args(parser, include_warmup=True)
    parser.add_argument(
        "--timestep_range",
        type=parse_float_tuple,
        default=(0.0, 1.0),
        metavar="LOW,HIGH",
        help=(
            "Sigma range for the flow-matching noise added to every training "
            "latent; sigma is drawn within [LOW, HIGH] per "
            "--timestep_distribution."
        ),
    )
    parser.add_argument(
        "--timestep_distribution",
        type=str,
        default="center",
        choices=["uniform", "center"],
        help=(
            "Sigma sampling distribution within timestep_range. "
            "center (default): Beta(2,2), peaks at the midpoint — the "
            "endpoints carry little learning signal. uniform: flat."
        ),
    )

    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help=(
            "Exponential moving average decay for trainable parameters "
            "(stitching layer + head suffix). 0.0 disables EMA. "
            "When set, an additional model_trainables_ema.pt is saved at every "
            "epoch end alongside the raw checkpoint."
        ),
    )

    parser.add_argument(
        "--token_weight",
        type=float,
        default=1.0,
        help="Weight for the token-level distillation loss.",
    )
    parser.add_argument(
        "--embed_weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the normalized embedding distillation loss. "
            "Only used when --feedforward_model=aesthetic."
        ),
    )
    parser.add_argument(
        "--score_weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the aesthetic score distillation loss. "
            "Only used when --feedforward_model=aesthetic."
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "l1"],
        help=(
            "Distillation loss type. Release: 'mse' everywhere except the "
            "SD3.5 Aesthetic models, which pass '--loss_type l1' "
            "(see the Stage 2 recipe in README.md)."
        ),
    )

    return parser
