# UnifiedReward-2.0 Qwen3.5 × SD3.5 Medium

[README](../README.md) · [Shared setup](SETUP.md) ·
[Stage 1 — Stitching-layer search](STAGE1_STITCHING_LAYER_SEARCH.md) ·
[Stage 2 — Value-model distillation](STAGE2_VALUE_MODEL_DISTILLATION.md) ·
[HPSv3](EXTENSION_HPSV3.md)

On this page: [Models](#supported-checkpoints) · [Setup](#setup) · [How to run](#how-to-run) ·
[Configuration](#configuration) · [Inference](#inference) · [Checkpoints](#checkpoints) ·
[Evaluation](#evaluation) · [Results](#results) ·
[Scope](#tested-scope) · [Source files](#implementation-file-map)

StitchVM connects SD3.5 Medium into the vision branch of the
[UnifiedReward-2.0](https://github.com/CodeGoat24/UnifiedReward) Qwen3.5
reward models (4B, 9B, 27B), which judge image–text pairs generatively. Only
the vision branch is distilled, and the language model is never constructed
during search or training. Evaluation runs the full original judge with the
stitched vision tokens in place of its own, on VL-RewardBench.

The [repository recipe](../README.md#training-recipe-and-relation-to-the-paper)
uses multi-resolution training and differs from the paper's setup.

## Supported checkpoints

Only these three checkpoints, at these revisions, have been tested. The
implementation reads every shape from the checkpoint's `config.json` and
`processor_config.json`, has no size-specific branch, and records the checkpoint
identity in its outputs. Other Qwen3.5 checkpoints load with a warning; Qwen3-VL
(DeepStack) and Qwen3.5-MoE towers are refused.

| Checkpoint | Revision | Vision tower | Merger output | Stitching width |
|---|---|---|---:|---:|
| `CodeGoat24/UnifiedReward-2.0-qwen35-4b` | `a1cb85ca` | 24 blocks, hidden 1024 | 2560 | 1024 |
| `CodeGoat24/UnifiedReward-2.0-qwen35-9b` | `f01548b0` | 27 blocks, hidden 1152 | 4096 | 1152 |
| `CodeGoat24/UnifiedReward-2.0-qwen35-27b` | `3a5c836f` | 27 blocks, hidden 1152 | 5120 | 1152 |

All three use 16-px patches (Conv3d 2×16×16), an interpolated learned absolute
position embedding added before block 0, 2D rotary embeddings, and one 2×2
PatchMerger. Their configs have `deepstack_visual_indexes = []` and contain no
DeepStack modules, so the language model receives only the merger output. That
output is the single distillation target.

The processor resizes each image to multiples of 32 px within
[65,536, 16,777,216] pixels, normalizes with mean = std = 0.5, and flattens
patches in 2×2 merge-window order. All five training buckets are already
multiples of 32 inside that range, so the reward input in search and training
is the VAE image itself. The head grid then equals the SD3.5 DiT token grid
(height/16 × width/16) and the post-stitch interpolation is the identity.

## Setup

Complete the environment installation in [shared setup](SETUP.md), then
download the checkpoints you need. HPDv2 and AVA are needed only for search
and training. Each is placed under `$STITCHVM_REWARD_CKPTS` with a
`PINNED_REVISION` file, which the identity guards read:

```bash
python scripts/download_weights.py stable_diffusion_3 unifiedreward_qwen35_9b
# also: unifiedreward_qwen35_4b (~10 GB), unifiedreward_qwen35_27b (~51 GB)
```

Use the download helper or a Hugging Face snapshot directory to retain the
immutable revision. Search, training and inference check this revision and the
config/processor hashes. Missing or incompatible identities are refused. Weight
files are not hashed; a manually written `PINNED_REVISION` is trusted.

Search and distillation read only the `model.visual.*` tensors. They use the
`qwen3_5` classes of the pinned Transformers 5.3.0, so no extra package is
needed. Evaluation additionally reads the benchmark parquet with `pyarrow`
([requirements-extensions.txt](../requirements-extensions.txt)):

```bash
python -m pip install -r requirements-extensions.txt
hf download MMInstruction/VL-RewardBench --repo-type dataset \
  --revision 0e6e62701eba92818a69ce95af0ed7aa0648b176 --local-dir "$DATA/vl_rewardbench"
# data/test-00000-of-00001.parquet: 1,247 items,
# SHA-256 500a33ecbeb6ddd4cbbb6de24b325815330c75c1f7b14e3043c1418fd285fb59
```

## How to run

```bash
bash scripts/stage1_find_stitching_layer.sh stage1_unifiedreward_qwen35_9b
DRY_RUN=1 NPROC=16 bash scripts/stage2_train_stitched_model.sh stage2_unifiedreward_qwen35_9b
NPROC=16 bash scripts/stage2_train_stitched_model.sh stage2_unifiedreward_qwen35_9b
```

The 4B and 27B recipes have the same names with `4b` / `27b`. Outputs:
`$INIT/unifiedreward_qwen35_<size>_sd3.5_dit_multires/` (search, including
`search_meta.json`) and
`$OUT/stitchvm_unifiedreward_qwen35_<size>_sd3/checkpoints/epoch_000{1..5}/model_trainables{,_ema}.pt`.

The search saves its model identity and configuration in `search_meta.json`.
Use a new search directory when changing the model or search settings. Training
requires a matching initialization and layer pair.

## Configuration

### Stage 1 — Stitching-layer search

Same protocol as the [release searches](STAGE1_STITCHING_LAYER_SEARCH.md#search-configuration):
- 300 HPDv2 + AVA images (200 fit / 100 validation);
- the five buckets 512², 768², 1024², 768×1024 and 1024×768;
- encoder outputs 1–4 × DiT outputs 1–12 (48 candidates);
- σ uniform on [0, 1];
- an affine 1×1 convolution to the vision hidden size, fitted by least squares.

The reward features are taken on the processor grid of the VAE image.

### Stage 2 — Value-model distillation

| Setting | Value |
|---|---|
| Launchers | `stage2_unifiedreward_qwen35_{4b,9b,27b}` |
| Layer pairs | 4B: DiT → encoder **4 → 2**; 9B and 27B: **3 → 2** (`enc_blocks_2_dit_4` / `enc_blocks_2_dit_3`) |
| Data / buckets | HPDv2 + AVA; the five buckets above; the shared paired augmentation |
| Excluded training images | 30 HPDv2 images ([list](../data_processing/vlrb_hpdv2_near_duplicates.txt)): near-duplicates of 6 VL-RewardBench test images |
| Teacher input | The processed VAE image at the checkpoint processor's size (native grid) |
| Loss | MSE over every element of the post-merger image tokens; no language-model forward |
| Trainable | Affine adapter, residual branch, vision blocks [2:], PatchMerger |
| Frozen | VAE, DiT prefix, teacher vision tower; the language model is never constructed |
| Optimizer | AdamW, LR 5e-5 (adapter multiplier 1), weight decay 0, 100 warmup updates, then constant; clipping 1.0 |
| Batch | Global 256 = micro-batch 8 × processes × accumulation |
| Noise | σ ~ Beta(2, 2) on [0, 1] (`center`) |
| EMA / length | Decay 0.999; five epochs (13,000 updates) |
| Precision | BF16 autocast, FP32 trainable copy, BF16 teacher; `--optimize_speed` |
| Memory | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by the launcher) |

The recipes exclude 30 HPDv2 images that are near-duplicates of six
VL-RewardBench test images. The [exclusion list](../data_processing/vlrb_hpdv2_near_duplicates.txt)
contains paths relative to the HPDv2 root; AVA is unchanged. Matching uses a
pHash/dHash screen (Hamming distance ≤ 10, including horizontal flips) followed
by grayscale-thumbnail correlation ≥ 0.95. Benchmark labels and answers are
not used. None of the excluded images is in the Stage 1 — Stitching-layer search samples.

To regenerate the list:

```bash
python data_processing/find_vlrb_near_duplicates.py \
  --vlrb_parquet "$DATA/vl_rewardbench/data/test-00000-of-00001.parquet" \
  --hpdv2_root "$DATA/hpdv2" --ava_root "$DATA/aesthetic_benchmark/ava" --out "$DATA/vlrb_dedup"
diff "$DATA/vlrb_dedup/exclude_paths.txt" data_processing/vlrb_hpdv2_near_duplicates.txt
```

## Inference

`StitchedUnifiedRewardJudge` ([stitchvm/models/unifiedreward_value_model.py](../stitchvm/models/unifiedreward_value_model.py))
loads a stitched checkpoint, the reward checkpoint and SD3.5 Medium, and generates the
judge's rationale and verdict for one image, one question and two candidate answers.
This interface compares two candidate answers; it does not generate a free-form
answer to the question. It uses the [evaluation protocol](#evaluation), including
the processor-native image size and a fixed σ and seed.

```bash
python scripts/infer_unifiedreward_judge.py \
  --checkpoint <dir>/model_trainables_ema.pt \
  --reward_checkpoint "$STITCHVM_REWARD_CKPTS/UnifiedReward-2.0-qwen35-9b" \
  --image photo.jpg --question "How many dogs are in the picture?" \
  --answer1 "There are two dogs." --answer2 "There is one cat." --sigma 0
```

```python
from stitchvm.models.unifiedreward_value_model import StitchedUnifiedRewardJudge
judge = StitchedUnifiedRewardJudge.from_pretrained(
    "<dir>/model_trainables_ema.pt", "<reward checkpoint dir>", sigma=0.0, seed=0)
result = judge.judge_pair("photo.jpg", question, answer1, answer2)
# result["output"]: generated text; result["verdict_official_phrases"]: 1, 2 or None;
# result["verdict_explicit_preference"]: 1, 2 or None
```

The stitched checkpoint must match the reward checkpoint's revision and
config/processor hashes. `--original_only` runs the unmodified judge.

Inference and evaluation pin SD3.5 Medium to
`b940f670f0eda2d07fbb75229e779da1ad11eb80` and reject incompatible export metadata.
Stage 1 — Stitching-layer search and Stage 2 — Value-model distillation use the
SD3.5 default branch; the reported weights use revision `b940f670`.

## Checkpoints

Epoch-5 EMA checkpoints are available on Hugging Face:

| Reward model | Checkpoint | Metadata and checksums |
|---|---|---|
| Qwen3.5 4B | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/unifiedreward_qwen35_4b_sd3.5_medium/model_trainables_ema.pt) | [Files](https://huggingface.co/HJGO/stitchvm/tree/main/unifiedreward_qwen35_4b_sd3.5_medium) |
| Qwen3.5 9B | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/unifiedreward_qwen35_9b_sd3.5_medium/model_trainables_ema.pt) | [Files](https://huggingface.co/HJGO/stitchvm/tree/main/unifiedreward_qwen35_9b_sd3.5_medium) |
| Qwen3.5 27B | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/unifiedreward_qwen35_27b_sd3.5_medium/model_trainables_ema.pt) | [Files](https://huggingface.co/HJGO/stitchvm/tree/main/unifiedreward_qwen35_27b_sd3.5_medium) |

The [shared inference example](../examples/README.md#unifiedreward) downloads the
selected StitchVM checkpoint automatically. To download it separately:

```bash
hf download HJGO/stitchvm unifiedreward_qwen35_9b_sd3.5_medium/model_trainables_ema.pt --local-dir checkpoints
```

A local training run writes `$OUT/stitchvm_unifiedreward_qwen35_<size>_sd3/checkpoints/epoch_0005/model_trainables_ema.pt`;
[scripts/export_extension_checkpoint.py](../scripts/export_extension_checkpoint.py) turns it
into a portable checkpoint that the inference and evaluation scripts load directly:

```bash
python scripts/export_extension_checkpoint.py --recipe stage2_unifiedreward_qwen35_9b \
  --checkpoint "$OUT/stitchvm_unifiedreward_qwen35_9b_sd3/checkpoints/epoch_0005/model_trainables_ema.pt" \
  --qwen35_checkpoint "$STITCHVM_REWARD_CKPTS/UnifiedReward-2.0-qwen35-9b" \
  --output_dir exports/unifiedreward_qwen35_9b_sd3.5_medium
```

The export contains FP32 EMA weights for the adapter, vision suffix and merger,
plus model revisions, preprocessing and recipe metadata. It writes
`model_trainables_ema.pt`, `stitchvm_export.json` and a model card. Frozen
models are loaded separately from their original sources.

## Evaluation

[scripts/eval_unifiedreward_vlrb.py](../scripts/eval_unifiedreward_vlrb.py)
generates the judge's verdict for every VL-RewardBench item. The original and
stitched runs use an identical protocol:
- the UnifiedReward pair-rank prompt, pinned to `CodeGoat24/UnifiedReward@ff6f5202`;
- the checkpoint's chat template with `enable_thinking=False`;
- the checkpoint processor applied once to the original image;
- greedy decoding with up to 4,096 new tokens;
- the official answer order (`random.seed(0)`, one draw per item).

The stitched run rebuilds each processed image from the processor output. It
encodes the image with the SD3.5 VAE, noises it at a fixed σ with seed 0, and
runs the DiT prefix and the stitched head on the image's native grid. The VAE
input is therefore the processor-native image: aspect ratio kept, both sides
multiples of 32 px, between 65,536 and 16,777,216 pixels (up to 16.6 megapixels
on VL-RewardBench). This differs from the fixed 1024 × 1024 VAE input of the
HPSv3 extension and of the twelve original pairs, so these scores are not a
1024² evaluation; a run with a pixel cap would be a different protocol. The
merged tokens replace the vision tower output; prompt placeholders, positions
and the language model are unchanged. If the VAE encode runs out of GPU memory
on the largest images, it retries once and then encodes on the CPU in FP32; the
device used is recorded per item.

```bash
CKPT=$STITCHVM_REWARD_CKPTS/UnifiedReward-2.0-qwen35-9b
PQ=$DATA/vl_rewardbench/data/test-00000-of-00001.parquet
EMA=$OUT/stitchvm_unifiedreward_qwen35_9b_sd3/checkpoints/epoch_0005/model_trainables_ema.pt
python scripts/eval_unifiedreward_vlrb.py --checkpoint $CKPT --vlrb_parquet $PQ \
  --items all --mode original --out results/vlrb_9b/original
for s in 0 0.1 0.25 0.5 0.75 0.9; do
  python scripts/eval_unifiedreward_vlrb.py --checkpoint $CKPT --vlrb_parquet $PQ \
    --items all --mode stitched --trainables $EMA --sigma $s --seed 0 --out results/vlrb_9b/sigma$s
done
python scripts/summarize_vlrb.py --original results/vlrb_9b/original \
  --stitched 0=results/vlrb_9b/sigma0 0.1=results/vlrb_9b/sigma0.1 0.25=results/vlrb_9b/sigma0.25 \
             0.5=results/vlrb_9b/sigma0.5 0.75=results/vlrb_9b/sigma0.75 0.9=results/vlrb_9b/sigma0.9 \
  --output results/vlrb_9b/summary.json
```

### Resume and sharding

Each run writes `run_meta.json` with its model, trainables SHA-256, dataset and
evaluation settings. Use a fresh output directory for a different configuration.

- Split a run with `--num_shards N --shard_index i` (one process per GPU).
- Continue with `--resume` and the same command. Changed settings or missing
  `run_meta.json` are refused.
- Merge with `scripts/merge_vlrb_shards.py OUT SHARD_DIRS...`. Every shard
  must have matching metadata and complete coverage of its assigned items.
- `summarize_vlrb.py` checks coverage, pairing, model identity, trainables and
  the labelled noise levels. To rescore outputs without `run_meta.json`, use
  `--legacy_inputs`; such outputs cannot be resumed or merged.

### Verdict parsing

The original and stitched models use the same parser:

- **Official:** the case-sensitive `Answer 1 is better` / `Answer 2 is better`
  substring rule from UnifiedReward. This is the primary benchmark score.
- **Strict:** accepts exactly one of these two phrases.
- **Explicit preference:** optional rescoring of clear candidate selections,
  including Markdown variants. Contradictions, ties, hypothetical choices and
  bare letters/numbers are left unrecognized.

The explicit-preference scores are reported separately from official scores.
Offline rescoring requires neither a GPU nor torch.

## Results

### Stage 1 — Stitching-layer search

One-based DiT → encoder outputs. MSE values are comparable only within the
same checkpoint's search.

| Checkpoint | Candidates | Selected DiT → encoder | Fit MSE | Validation MSE |
|---|---:|---:|---:|---:|
| 4B | 48 | 4 → 2 | 0.028991 | 0.034251 |
| 9B | 48 | 3 → 2 | 0.020946 | 0.021973 |
| 27B | 48 | 3 → 2 | 0.022547 | 0.023771 |

### Stage 2 — Value-model distillation

VL-RewardBench accuracy (%), all 1,247 items, epoch-5 EMA weights. Each row
reports the complete noise sweep. Both parsers score the same generated answers.

**Official parser**

| Model | Original | σ=0 | σ=0.1 | σ=0.25 | σ=0.5 | σ=0.75 | σ=0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 75.30 | 72.41 | 73.14 | 72.41 | 72.09 | 70.73 | 69.69 |
| 9B | 84.36 | 82.92 | 82.68 | 83.56 | 81.96 | 79.47 | 79.31 |
| 27B | 80.91 | 80.27 | 80.43 | 80.99 | 79.79 | 76.02 | 73.78 |

**Explicit-preference parser**

| Model | Original | σ=0 | σ=0.1 | σ=0.25 | σ=0.5 | σ=0.75 | σ=0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 76.98 | 74.42 | 74.90 | 74.18 | 73.14 | 71.85 | 70.97 |
| 9B | 84.44 | 83.08 | 82.76 | 83.48 | 82.04 | 79.63 | 79.55 |
| 27B | 83.64 | 83.00 | 83.72 | 83.48 | 81.88 | 77.87 | 75.78 |

[Detailed results](UNIFIEDREWARD_QWEN35_RESULTS.md) include invalid counts,
category averages, verdict agreement and paired comparisons. In the 27B sweep,
the largest image (16.6 MP) uses the CPU VAE fallback at each noise level.

## Tested scope

Supported: SD3.5 Medium with the three checkpoints above and Transformers 5.3.0.
Other diffusion backbones, Qwen3-VL and Qwen3.5-MoE are not supported. Other dense
Qwen3.5 checkpoints are unvalidated and load with a warning.

Evaluation uses the checkpoint's processor without the extra pre-resize step of
the Qwen2.5-VL evaluation script. Scores apply to the protocol documented here.

## Implementation file map

| File | Role |
|---|---|
| [stitchvm/models/feedforward/qwen35_vision_head.py](../stitchvm/models/feedforward/qwen35_vision_head.py) | Processor policy, preprocessing and inverse, vision suffix head (official rotary function, FP32 rotary table), teacher wrapper, vision-only loader, checkpoint identity and search-sidecar guard. |
| [stitchvm/models/feedforward/qwen2vl_vision_head.py](../stitchvm/models/feedforward/qwen2vl_vision_head.py) | Shared merge-window token order, patchification and search hooks. |
| [stitchvm/losses/qwen2vl_loss.py](../stitchvm/losses/qwen2vl_loss.py) | Post-merger image-token MSE. |
| [find_stitching_layer.py](../find_stitching_layer.py), [train_stitched_model.py](../train_stitched_model.py) | `qwen35` head in search and distillation; `search_meta.json`; `reward_meta`. |
| [stitchvm/data/dataset_util.py](../stitchvm/data/dataset_util.py) | `--exclude_image_list`. |
| [scripts/stage1_find_stitching_layer.sh](../scripts/stage1_find_stitching_layer.sh), [scripts/stage2_train_stitched_model.sh](../scripts/stage2_train_stitched_model.sh), [scripts/stage2_config.sh](../scripts/stage2_config.sh) | `stage1_unifiedreward_qwen35_*`, `stage2_unifiedreward_qwen35_*`. |
| [stitchvm/models/qwen35_judge.py](../stitchvm/models/qwen35_judge.py) | Judge wrapper, stitched / identity-injection vision branches. |
| [stitchvm/utils/vlrb_protocol.py](../stitchvm/utils/vlrb_protocol.py) | Pinned prompt, official answer order, category mappings, official and strict parsers, metrics. |
| [stitchvm/utils/vlrb_run_meta.py](../stitchvm/utils/vlrb_run_meta.py) | Run provenance (`run_meta.json`, run id) checked by resume, merge and summary. |
| [stitchvm/utils/explicit_preference.py](../stitchvm/utils/explicit_preference.py) | Optional explicit-preference parser. |
| [stitchvm/models/unifiedreward_value_model.py](../stitchvm/models/unifiedreward_value_model.py), [scripts/infer_unifiedreward_judge.py](../scripts/infer_unifiedreward_judge.py) | Stitched pairwise-judge inference and its CLI (also used by the evaluation). |
| [stitchvm/utils/extension_checkpoint.py](../stitchvm/utils/extension_checkpoint.py), [scripts/export_extension_checkpoint.py](../scripts/export_extension_checkpoint.py) | Portable checkpoint format, weights-only loader, export. |
| [scripts/eval_unifiedreward_vlrb.py](../scripts/eval_unifiedreward_vlrb.py), [scripts/merge_vlrb_shards.py](../scripts/merge_vlrb_shards.py), [scripts/summarize_vlrb.py](../scripts/summarize_vlrb.py) | Generation, shard validation and merge, validated summary and rescoring. |
| [data_processing/vlrb_hpdv2_near_duplicates.txt](../data_processing/vlrb_hpdv2_near_duplicates.txt), [data_processing/find_vlrb_near_duplicates.py](../data_processing/find_vlrb_near_duplicates.py) | Excluded training images and the script that finds them. |
