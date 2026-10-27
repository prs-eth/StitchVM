# Shared setup

[README](../README.md) · [Stage 1 — Stitching-layer search](STAGE1_STITCHING_LAYER_SEARCH.md) ·
[Stage 2 — Value-model distillation](STAGE2_VALUE_MODEL_DISTILLATION.md) ·
Model guides: [HPSv3](EXTENSION_HPSV3.md), [UnifiedReward-2.0 Qwen3.5](EXTENSION_UNIFIEDREWARD_QWEN35.md)

This guide prepares the environment, pretrained dependencies, and images used
by both workflows. Run all commands from the repository root.

The repository's training recipe differs from the paper: multi-resolution
training was introduced to improve support across image resolutions. The
workflow guides document the current executable configuration.
Pretrained EMA checkpoints of the twelve original backbone–reward pairs are
[published](../README.md#pretrained-checkpoints), along with
[HPSv3 and UnifiedReward-2.0 Qwen3.5 checkpoints](../README.md#vision-language-reward-models).
For image scoring, follow the [inference examples](../examples/README.md).
To train, generate the stitching
initializations locally; downloads below provide upstream pretrained
dependencies.

## Install

Use Python 3.11–3.12 and PyTorch 2.7. Full searches and training require CUDA
with BF16 support. Search runs in one process on one GPU. Distillation supports
one process or distributed data parallelism through Accelerate. Each training
process loads its own backbone, teacher, adapter, and suffix; increasing the
process count does not shard these models across devices.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# CUDA 12.8 example; choose a PyTorch 2.7 wheel compatible with your driver.
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e ./external_repos/HPSv2
python -m pip install -e .
python -m pip check
python find_stitching_layer.py --help
python train_stitched_model.py --help
```

Pinned dependencies include Transformers 5.3.0, Diffusers 0.37.1, Accelerate
1.13.0, and open_clip_torch 2.32.0. See [requirements.txt](../requirements.txt)
for the full direct requirements.

## Configure paths

```bash
export DATA="$PWD/data"
export INIT="$PWD/init"
export OUT="$PWD/outputs"
export HF_HOME="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"
export STITCHVM_REWARD_CKPTS="$PWD/reward_ckpts"
```

Defaults are in [scripts/paths.sh](../scripts/paths.sh). Use absolute paths to
reuse datasets and caches outside the checkout. `NUM_WORKERS` defaults to `3`
and `PREFETCH_FACTOR` to `2` per process; `NUM_WORKERS=0` disables worker
prefetch and persistence. `STITCHVM_VAE_SLICING=1` encodes images individually
to reduce VAE batch memory. Plan for the largest bucket and model-loading peak.

## Download pretrained dependencies

Accept the applicable model-access conditions and authenticate with
`hf auth login` before downloading gated backbones. Original model and dataset
licenses apply. List the download names, then fetch the components you need:

```bash
python scripts/download_weights.py --list
# Example: SD3.5 Medium with CLIP.
python scripts/download_weights.py stable_diffusion_3 clip
# All supported backbones and reward dependencies, if needed.
python scripts/download_weights.py stable_diffusion_3 stable_diffusion_3_large flux_dev
python scripts/download_weights.py hpsv2 clip dfn_clip aesthetic
```

| Component | Pretrained source | Dimensions |
|---|---|---|
| SD3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` | 24 DiT blocks |
| SD3.5 Large | `stabilityai/stable-diffusion-3.5-large` | 38 DiT blocks |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | 19 dual-stream + 38 single-stream blocks |
| HPSv2 | ViT-H/14 base + `HPS_v2.1_compressed.pt` | Input 224²; patch grid 16×16; hidden 1280; embedding 1024 |
| CLIP | `openai/clip-vit-large-patch14-336` | Input 336²; patch grid 24×24; hidden 1024; embedding 768 |
| Aesthetic encoder | `openai/clip-vit-large-patch14` | Input 224²; patch grid 16×16; hidden 1024; embedding 768 |
| DFN-CLIP | `hf-hub:apple/DFN5B-CLIP-ViT-H-14-378` | Input 378²; patch grid 27×27; hidden 1280; embedding 1024 |

The vision-language reward models (`hpsv3`, `unifiedreward_qwen35_{4b,9b,27b}`) are
downloaded by name only, not by `--all`, at pinned revisions; their guides list
the additional evaluation dependencies.

The `clip` download includes both 224px and 336px encoders. HPSv2 additionally
uses two files under the reward-checkpoint directory:

```text
$STITCHVM_REWARD_CKPTS/hpsv2/
├── open_clip_pytorch_model.bin
└── HPS_v2.1_compressed.pt
```

Aesthetic distillation needs the frozen scoring MLP at
`$STITCHVM_REWARD_CKPTS/sac+logos+ava1-l14-linearMSE.pth`, downloaded by the
`aesthetic` command. The feature search uses only its CLIP encoder. Text encoders
and tokenizers are needed to compute empty-prompt conditioning even though
image captions are not used.

Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` only after all needed model and
tokenizer files are cached. The original twelve recipes load the upstream models'
default revisions. HPSv3 and UnifiedReward use pinned reward-model revisions;
their inference loaders also pin SD3.5 Medium, as documented in the model guides.
For reproducible local runs, retain the source revision, installed package
versions, pretrained revisions, and dataset inventories with your run.

## Prepare HPDv2 and AVA

All twelve model pairs use **both HPDv2 and AVA training images** for search and
distillation. Labels and preference scores are not training targets; the frozen
reward encoder provides the targets.

### HPDv2

```bash
python data_processing/prepare_hpdv2_train.py --output_dir "$DATA/hpdv2" --skip_json
```

Allow roughly 62 GB during extraction for the archive and images. The script
removes the archive afterward unless `--keep_tar` is set. Metadata is optional
for these workflows because they read unique images directly from `train/`.

### AVA

Download the JPEGs from the
[AVA Kaggle dataset](https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment)
and extract them into `$DATA/aesthetic_benchmark/ava/images/`, without an extra
nested image folder. Then prepare the metadata:

```bash
python data_processing/prepare_aesthetic_benchmark.py ava \
  --output_dir "$DATA/aesthetic_benchmark/ava"
```

Expected layout:

```text
$DATA/
├── hpdv2/
│   └── train/*.jpg
└── aesthetic_benchmark/ava/
    ├── AVA.txt
    └── images/*.jpg
```

The AVA loader parses valid, nonzero-rating rows, reserves the last 20,000 rows
for the held-out split, and filters the preceding training IDs to images present
on disk. Use the complete training image set and check that both datasets load
with the expected image counts; missing files change the training population.
The held-out split depends on the contents and ordering of `AVA.txt`.

The datasets are concatenated and randomly sampled, without 50/50 balancing,
per-dataset reweighting, or per-bucket quotas. The loader tolerates truncated
images and, on a loading failure, tries up to ten consecutive indices with
wrapping, substituting the first image that loads.

## Multi-resolution inputs and preprocessing

Multi-resolution training improves support across image sizes. Search and
distillation use the same buckets and datasets; their augmentation and noise
sampling differ as specified in the workflow guides. Dimensions below are
**width × height**.

| Setting | HPSv2 / CLIP / Aesthetic | DFN-CLIP |
|---|---|---|
| VAE input buckets | 512², 768², 1024², 768×1024, 1024×768 | 512², 768², 1024² |
| Reward input | Encoder's native square size in the model table | 378² |
| Bucket sampling | Uniform per batch, independent of image identity | Same |
| Loader tail | Drop incomplete image micro-batches | Same |

Images are RGB. Search resizes directly to the chosen bucket without random
crop or color jitter, stretching to its dimensions. Distillation resizes to
bucket dimensions multiplied by a random scale in [1.1, 1.3], crops back to the
bucket, and applies color jitter (brightness/contrast/saturation 0.5, hue 0.1).
The reward view is resized from the same processed VAE view; rectangular inputs
are stretched to the reward encoder's square size. Both views are represented
in [−1, 1]; reward input is mapped back to [0, 1] and normalized with its
encoder's mean and standard deviation.

Continue with [Stage 1 — Stitching-layer search](STAGE1_STITCHING_LAYER_SEARCH.md), then
[Stage 2 — Value-model distillation](STAGE2_VALUE_MODEL_DISTILLATION.md).
