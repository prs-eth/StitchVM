# HPSv3 × SD3.5 Medium

[README](../README.md) · [Shared setup](SETUP.md) ·
[Stage 1 — Stitching-layer search](STAGE1_STITCHING_LAYER_SEARCH.md) ·
[Stage 2 — Value-model distillation](STAGE2_VALUE_MODEL_DISTILLATION.md) ·
[UnifiedReward-2.0 Qwen3.5](EXTENSION_UNIFIEDREWARD_QWEN35.md)

On this page: [Model](#model) · [Setup](#setup) · [How to run](#how-to-run) ·
[Configuration](#configuration) · [Inference](#inference) · [Checkpoints](#checkpoints) ·
[Evaluation](#evaluation) · [Results](#results) ·
[Scope](#tested-scope) · [Source files](#implementation-file-map)

StitchVM connects SD3.5 Medium into the vision branch of
[HPSv3](https://github.com/MizzenAI/HPSv3), a vision-language preference model.
Only the vision branch is distilled; the language model and reward head are
frozen and are not loaded for search or training. Evaluation runs the full
original HPSv3 model with the stitched vision tokens in place of its own.

The [repository recipe](../README.md#training-recipe-and-relation-to-the-paper)
uses multi-resolution training and differs from the paper's setup.

## Model

| Component | Value |
|---|---|
| Reward model | `MizzenAI/HPSv3` (`HPSv3.safetensors`, revision `4f81e3e0`) on `Qwen/Qwen2-VL-7B-Instruct` (revision `eed13092`) |
| Vision tower | Qwen2-VL: 14-px patches (Conv3d 2×14×14), 32 blocks, hidden 1280, 2D rotary embeddings, 2×2 PatchMerger to 3584 |
| Reward | Language model + ranknet head at the `<|Reward|>` token; the reward is μ of the predicted (μ, σ) |
| Stitched head | Vision blocks [N:] + PatchMerger; output: the merged image tokens the language model consumes |
| Teacher input in training | 448 × 448 (32 × 32 patch grid, 256 merged tokens) |

The head uses the native 2×2 merge-window token order and recomputes rotary
positions and attention segments for each image grid.

## Setup

Complete the environment installation in [shared setup](SETUP.md). HPDv2 and
AVA are needed only for search and training. Download the upstream models next. The `hpsv3` entry fetches the HPSv3 checkpoint and its Qwen2-VL base at
the pinned revisions into the Hugging Face cache (~32 GB):

```bash
python scripts/download_weights.py stable_diffusion_3 hpsv3
```

Search and distillation read only the `visual.*` tensors of `HPSv3.safetensors`
plus the vision configuration of the base model. `--hpsv3_checkpoint` accepts a
local copy of the file.

Install the inference and evaluation dependencies, including the
[HPSv3 input processor](../external_repos/HPSv3/VENDORED.md):

```bash
python -m pip install -r requirements-extensions.txt
# Optional: the reported evaluation used FlashAttention-2 for the reward model.
python -m pip install flash-attn --no-build-isolation
```

## How to run

```bash
bash scripts/stage1_find_stitching_layer.sh stage1_hpsv3
DRY_RUN=1 NPROC=16 bash scripts/stage2_train_stitched_model.sh stage2_hpsv3
NPROC=16 bash scripts/stage2_train_stitched_model.sh stage2_hpsv3
```

The launcher derives gradient accumulation from `NPROC` and the micro-batch
(global batch 256, micro-batch 8; for example 16 processes × 8 × 2). Multi-node
settings follow the [distillation guide](STAGE2_VALUE_MODEL_DISTILLATION.md#batch-size-and-memory).
Outputs follow the existing conventions:
`$INIT/hpsv3_sd3.5_dit_multires/` for the search and
`$OUT/stitchvm_hpsv3_sd3/checkpoints/epoch_000{1..5}/model_trainables{,_ema}.pt`
for distillation.

## Configuration

### Stage 1 — Stitching-layer search

Same protocol as the [twelve release searches](STAGE1_STITCHING_LAYER_SEARCH.md#search-configuration):
300 HPDv2 + AVA images (200 fit / 100 validation), the five resolution buckets
512², 768², 1024², 768×1024 and 1024×768, encoder outputs 1–4 × DiT outputs
1–12 (48 candidates), σ uniform on [0, 1], and an affine 1×1 convolution to 1280
channels fitted by least squares. The reward input is the processed image
resized to 448², and features are compared on the 32 × 32 patch grid.

### Stage 2 — Value-model distillation

| Setting | Value |
|---|---|
| Launcher | `stage2_hpsv3` |
| Layer pair | DiT → encoder **5 → 2** (`enc_blocks_2_dit_5`) |
| Data / buckets | HPDv2 + AVA; the five buckets above; the shared paired augmentation |
| Loss | MSE over the post-merger image tokens (B × 256 × 3584) |
| Trainable | Affine adapter, residual branch, vision blocks 2–31, PatchMerger (637.3 M parameters) |
| Frozen | VAE, DiT prefix, teacher vision tower; the language branch is never constructed |
| Optimizer | AdamW, LR 5e-5 (adapter multiplier 1), weight decay 0, 100 warmup updates, then constant; clipping 1.0 |
| Batch | Global 256 = micro-batch 8 × processes × accumulation |
| Noise | σ ~ Beta(2, 2) on [0, 1] (`center`) |
| EMA / length | Decay 0.999; five epochs (2,600 updates per epoch on the full HPDv2 + AVA set) |
| Precision | BF16 autocast, FP32 trainable copy, BF16 teacher; `--optimize_speed` |
| Memory | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by the launcher) |

Allow for about 45 GB of steady-state GPU memory and up to 91 GB during cuDNN
bucket benchmarking. `STITCHVM_VAE_SLICING=1` lowers the VAE memory peak.

## Inference

`StitchedHPSv3` ([stitchvm/models/hpsv3_value_model.py](../stitchvm/models/hpsv3_value_model.py)) loads a
stitched checkpoint with SD3.5 Medium and the full HPSv3 model, and returns HPSv3 rewards
(μ) computed from SD3.5 latents:
- `score(images, prompts, sigma, seed)`: each image is resized to 1024² for the VAE,
  noised as z_σ = (1 − σ) z + σ ε with a seeded generator, and scored with the stitched
  head on the image's native HPSv3 grid;
- `score_noisy_latents(latents, sigma, prompts)`: SD3.5 latents that are already noised to
  σ, for example inside a sampler; the head grid is the HPSv3 grid of the decoded size;
- `score_original(images, prompts)`: the unmodified HPSv3 model.

```bash
python scripts/infer_hpsv3.py --checkpoint <dir>/model_trainables_ema.pt \
  --prompt "a red bicycle leaning against a brick wall" --image a.png b.png --sigma 0.25 --original
```

```python
from stitchvm.models.hpsv3_value_model import StitchedHPSv3
model = StitchedHPSv3.from_pretrained("<dir>/model_trainables_ema.pt")
rewards = model.score(["a.png", "b.png"], [prompt, prompt], sigma=0.25, seed=0)
```

Portable checkpoints verify the HPSv3 file's SHA-256, including local files
passed with `--hpsv3_checkpoint`. Training checkpoints lack this identity and
load with a warning.

Inference and evaluation pin SD3.5 Medium to
`b940f670f0eda2d07fbb75229e779da1ad11eb80` and reject incompatible export metadata.
Stage 1 — Stitching-layer search and Stage 2 — Value-model distillation use the
SD3.5 default branch; the reported weights use revision `b940f670`.

## Checkpoints

The [epoch-5 EMA checkpoint](https://huggingface.co/HJGO/stitchvm/resolve/main/hpsv3_sd3.5_medium/model_trainables_ema.pt)
is available on Hugging Face, with [metadata and checksums](https://huggingface.co/HJGO/stitchvm/tree/main/hpsv3_sd3.5_medium).
The [shared inference example](../examples/README.md#hpsv3) downloads it automatically.

```bash
hf download HJGO/stitchvm hpsv3_sd3.5_medium/model_trainables_ema.pt --local-dir checkpoints
```

A local training run writes `$OUT/stitchvm_hpsv3_sd3/checkpoints/epoch_0005/model_trainables_ema.pt`;
[scripts/export_extension_checkpoint.py](../scripts/export_extension_checkpoint.py) turns it
into a portable checkpoint that the inference and evaluation scripts load directly:

```bash
python scripts/export_extension_checkpoint.py --recipe stage2_hpsv3 \
  --checkpoint "$OUT/stitchvm_hpsv3_sd3/checkpoints/epoch_0005/model_trainables_ema.pt" \
  --output_dir exports/hpsv3_sd3.5_medium
```

The export contains FP32 EMA weights for the adapter, vision suffix and merger,
plus model revisions, preprocessing and recipe metadata. It writes
`model_trainables_ema.pt`, `stitchvm_export.json` and a model card. Frozen
models are loaded separately from their original sources.

## Evaluation

[scripts/eval_hpsv3_benchmark.py](../scripts/eval_hpsv3_benchmark.py) scores
the original HPSv3 model and the stitched model with the same language branch,
prompt and reward head:

- **Original**: inputs are built exactly as in HPSv3's inferencer (prompt
  template, 256 × 28 × 28 pixel budget).
- **Stitched**: each image is resized bicubically to 1024² for the VAE and
  noised at a fixed σ with a fixed seed per noise level. The stitched head is
  interpolated to that image's native HPSv3 patch grid, so the language model
  sees the same token layout as for the original model.

The benchmark datasets are pinned to `ymhao/HPDv2@bb0f2563` and `THUDM/ImageRewardDB@c493721e`.

```bash
python data_processing/prepare_hpdv2_test.py --output_dir "$DATA/hpsv3_benchmark/hpdv2"
python data_processing/prepare_imagereward_benchmark.py --output_dir "$DATA/hpsv3_benchmark/imagereward"
# Optional compatibility set (JPEG quality 95 re-encoding):
python data_processing/prepare_imagereward_benchmark.py --format jpeg95 \
  --output_dir "$DATA/hpsv3_benchmark/imagereward_jpeg95"

CKPT="$OUT/stitchvm_hpsv3_sd3/checkpoints/epoch_0005/model_trainables.pt"
python scripts/eval_hpsv3_benchmark.py --checkpoint_path "$CKPT" --use_ema \
  --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
  --stitching_layer_location enc_blocks_2_dit_5 --vae_input_size 1024 \
  --dataset hpdv2 --data_path "$DATA/hpsv3_benchmark/hpdv2" \
  --image_path "$DATA/hpsv3_benchmark/hpdv2/test" \
  --noise_levels 0.1 0.25 0.5 0.75 0.9 --eval_original --output results/hpsv3_sd3_hpdv2.json
python scripts/eval_hpsv3_benchmark.py --checkpoint_path "$CKPT" --use_ema \
  --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
  --stitching_layer_location enc_blocks_2_dit_5 --vae_input_size 1024 \
  --dataset imagereward --data_path "$DATA/hpsv3_benchmark/imagereward" \
  --image_path "$DATA/hpsv3_benchmark/imagereward/images" \
  --noise_levels 0.1 0.25 0.5 0.75 0.9 --eval_original --output results/hpsv3_sd3_imagereward.json
```

The result file records the stitched checkpoint actually loaded under `trainables` (content
SHA-256, portable export or training checkpoint, EMA status; no local path). `use_ema_flag`
only records whether `--use_ema` selected `model_trainables_ema.pt` next to `--checkpoint_path`.

Metrics, as percentages averaged over prompts:
- **HPDv2 test**: inversion-score accuracy of the predicted ranking (400 prompts).
- **ImageReward test**: pairwise accuracy with ties excluded (466 prompts).
- **Pooled pair accuracy**: HPSv3's own `calc_rank_acc`, reported alongside as `hpsv3_rank_acc_pct`.

ImageReward uses the original WebP images as the primary input; the JPEG-q95
re-encoding is a separate compatibility number.

## Results

Stage 1 — Stitching-layer search selected DiT → encoder **5 → 2**
(`enc_blocks_2_dit_5`): fit MSE **0.18268**, validation MSE **0.20928**.
MSE values are comparable only within the same reward encoder.

Stage 2 — Value-model distillation results use epoch-5 EMA weights. The
original and stitched models use the same evaluation protocol. The HPSv3 paper reports
85.4% (HPDv2) and 66.8% (ImageReward) for the original model. All evaluated
noise levels are shown; σ = 0 was not evaluated for this model.

| Benchmark (metric) | Original | σ=0.1 | σ=0.25 | σ=0.5 | σ=0.75 | σ=0.9 |
|---|---:|---:|---:|---:|---:|---:|
| HPDv2 test (inversion-score accuracy) | 85.45 | 85.00 | 84.88 | 84.45 | 80.91 | 75.73 |
| ImageReward, original WebP (pairwise accuracy) | 67.77 | 68.20 | 67.79 | 67.58 | 64.18 | 59.73 |
| ImageReward, JPEG-q95 (pairwise accuracy) | 66.60 | 68.73 | 68.11 | 67.98 | 64.83 | 60.43 |
| HPDv2, HPSv3 pooled pair accuracy | 85.37 | 84.96 | 84.84 | 84.41 | 80.84 | 75.59 |
| ImageReward WebP, HPSv3 pooled pair accuracy | 66.92 | 67.28 | 66.73 | 66.95 | 63.96 | 59.62 |
| ImageReward JPEG-q95, HPSv3 pooled pair accuracy | 65.96 | 67.62 | 66.82 | 67.28 | 64.42 | 60.06 |

These results are from one training run without repeat-variability estimates.

## Tested scope

Supported: SD3.5 Medium with the HPSv3 checkpoint above and Transformers 5.3.0.
Other backbones are not supported by this head. The reported evaluation uses
FlashAttention-2; the default SDPA backend may give small numerical differences.

## Implementation file map

| File | Role |
|---|---|
| [stitchvm/models/feedforward/qwen2vl_vision_head.py](../stitchvm/models/feedforward/qwen2vl_vision_head.py) | Qwen2-VL patchification, merge-window token order, rotary positions, vision suffix head, teacher wrapper, search hooks. |
| [stitchvm/losses/qwen2vl_loss.py](../stitchvm/losses/qwen2vl_loss.py) | Post-merger image-token MSE. |
| [stitchvm/utils/utils_for_thirdparty.py](../stitchvm/utils/utils_for_thirdparty.py) | `load_hpsv3_visual`: vision tower only, from the pinned checkpoint. |
| [stitchvm/models/qwen2vl_reward.py](../stitchvm/models/qwen2vl_reward.py) | Full HPSv3 reward model for evaluation (official forward; swappable vision tokens). |
| [stitchvm/models/stitched_image_model.py](../stitchvm/models/stitched_image_model.py) | `target_grid` for per-image head grids. |
| [find_stitching_layer.py](../find_stitching_layer.py), [train_stitched_model.py](../train_stitched_model.py) | `hpsv3` head in search and distillation. |
| [scripts/stage1_find_stitching_layer.sh](../scripts/stage1_find_stitching_layer.sh), [scripts/stage2_train_stitched_model.sh](../scripts/stage2_train_stitched_model.sh), [scripts/stage2_config.sh](../scripts/stage2_config.sh) | `stage1_hpsv3`, `stage2_hpsv3`. |
| [stitchvm/models/hpsv3_value_model.py](../stitchvm/models/hpsv3_value_model.py), [scripts/infer_hpsv3.py](../scripts/infer_hpsv3.py) | Stitched HPSv3 inference (images or noisy latents) and its CLI. |
| [stitchvm/utils/extension_checkpoint.py](../stitchvm/utils/extension_checkpoint.py), [scripts/export_extension_checkpoint.py](../scripts/export_extension_checkpoint.py) | Portable checkpoint format, weights-only loader, export. |
| [scripts/eval_hpsv3_benchmark.py](../scripts/eval_hpsv3_benchmark.py) | HPDv2 / ImageReward evaluation of the original and stitched models (uses the inference module). |
| [data_processing/prepare_hpdv2_test.py](../data_processing/prepare_hpdv2_test.py), [data_processing/prepare_imagereward_benchmark.py](../data_processing/prepare_imagereward_benchmark.py) | Benchmark preparation. |
| [external_repos/HPSv3](../external_repos/HPSv3/VENDORED.md) | Vendored HPSv3 input processing and prompt text (MIT). |
