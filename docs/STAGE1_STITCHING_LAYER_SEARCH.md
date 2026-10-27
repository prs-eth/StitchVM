# Stage 1 — Stitching-layer search

[README](../README.md) · [Shared setup](SETUP.md) ·
[Stage 2 — Value-model distillation](STAGE2_VALUE_MODEL_DISTILLATION.md) ·
Model guides: [HPSv3](EXTENSION_HPSV3.md) · [UnifiedReward-2.0 Qwen3.5](EXTENSION_UNIFIEDREWARD_QWEN35.md)

On this page: [Setup](#setup) · [How to run](#how-to-run) ·
[Configuration](#search-configuration) · [Results](#results) ·
[Source files](#implementation-file-map)

This workflow finds where to connect a frozen diffusion backbone to a pretrained
reward encoder. It extracts paired spatial features, fits an affine 1×1
convolution for each candidate block pair, and saves fitting and held-out MSE.
The fitted adapter initializes subsequent value-model distillation.

The stage number is a repository workflow label. This guide describes the
current multi-resolution search configuration; the repository's training recipe
differs from the paper, as explained in the [README](../README.md#training-recipe-and-relation-to-the-paper).

## Setup

Complete [shared setup](SETUP.md): install dependencies, configure paths, download
the chosen diffusion backbone and reward encoder, and prepare **both HPDv2 and
AVA**. Use a CUDA GPU with BF16 support. The search runs on one GPU in one
process and requires substantial CPU RAM and disk for cached features.
Precomputed stitching initializations are not included; this workflow generates
them locally.

## How to run

List the available searches, then run one or more sequentially:

```bash
bash scripts/stage1_find_stitching_layer.sh
bash scripts/stage1_find_stitching_layer.sh stage1_clip
# Example: DFN-CLIP for all three backbones.
bash scripts/stage1_find_stitching_layer.sh \
  stage1_dfn_clip stage1_large_dfn_clip stage1_flux_dfn_clip
```

| Backbone | Encoder | Launcher name | Output directory under `$INIT` |
|---|---|---|---|
| SD3.5 Medium | HPSv2 | `stage1_hpsv2` | `hpsv2_sd3.5_dit_multires` |
| SD3.5 Medium | CLIP | `stage1_clip` | `clip_sd3.5_dit_multires` |
| SD3.5 Medium | Aesthetic | `stage1_aesthetic` | `aesthetic_sd3.5_dit_multires` |
| SD3.5 Medium | DFN-CLIP | `stage1_dfn_clip` | `dfn_clip_sd3.5_dit_multires` |
| SD3.5 Large | HPSv2 | `stage1_large_hpsv2` | `hpsv2_sd3.5_large_dit_multires` |
| SD3.5 Large | CLIP | `stage1_large_clip` | `clip_sd3.5_large_dit_multires` |
| SD3.5 Large | Aesthetic | `stage1_large_aesthetic` | `aesthetic_sd3.5_large_dit_multires` |
| SD3.5 Large | DFN-CLIP | `stage1_large_dfn_clip` | `dfn_clip_sd3.5_large_dit_multires` |
| FLUX.1-dev | HPSv2 | `stage1_flux_hpsv2` | `hpsv2_flux_dev_dit_multires` |
| FLUX.1-dev | CLIP | `stage1_flux_clip` | `clip_flux_dev_dit_multires` |
| FLUX.1-dev | Aesthetic | `stage1_flux_aesthetic` | `aesthetic_flux_dev_dit_multires` |
| FLUX.1-dev | DFN-CLIP | `stage1_flux_dfn_clip` | `dfn_clip_flux_dev_dit_multires` |

Use the named shell recipes to obtain the complete configuration; the generic
Python CLI defaults do not select one of these recipes automatically.

## Search configuration

All pairs use the same 300-image budget, split, and candidate-depth rule. Search
uses the [same resolution buckets as distillation](SETUP.md#multi-resolution-inputs-and-preprocessing):
five buckets for HPSv2, CLIP, and Aesthetic; three square buckets for DFN-CLIP.
The reward view stays at its encoder's native size. Augmentation is disabled.

| Setting | Value for every backbone/head pair |
|---|---|
| Images / batch size | 300 total / 1 |
| Fit / validation | First 200 / last 100 of the sampled sequence |
| Validation fraction | 0.3334; validation count is `max(1, int(n × fraction))` |
| Candidate encoder outputs | Blocks 1–4 |
| Candidate DiT outputs | First half, rounded down: Medium 1–12, Large 1–19, FLUX 1–28 |
| Candidate count | Medium 48, Large 76, FLUX 112 per encoder |
| Conditioning | Empty prompt; FLUX guidance embedding scalar 3.5 |
| VAE latent | Posterior sample, then `(latent − shift_factor) × scaling_factor` |
| Noise | Independent Gaussian noise; σ uniform on [0, 1]; `zσ = (1−σ)z + σε` |
| Feature extraction precision | BF16 autocast; cached spatial features FP16 on CPU |
| Fitted adapter | Affine 1×1 convolution, stride 1, padding 0, bias enabled |
| Fit precision / solver | Float64 joint weight-and-bias `torch.linalg.lstsq` |
| Spatial alignment | Bilinear interpolation to the reward patch grid, `align_corners=True` |
| Selection metric | Lowest held-out MSE over all patch-token feature elements; class token excluded |
| Seed | No fixed extraction seed in the launcher |

The 300-image budget spans both datasets and all buckets; it is not 300 per
bucket or dataset. Prepare enough images for the complete budget. All candidate
pairs within a search share the sampled features and split. Restricting the
search to these depths defines the candidate space; omitted depths are not
assigned scores. CUDA's least-squares solver assumes full column rank.

Block numbers refer to **one-based outputs**. FLUX counts dual-stream blocks
1–19, followed by single-stream blocks 20–57, so its first-half search covers
outputs 1–28. Reward output N feeds the suffix starting at zero-based block N.

## Outputs and choosing a pair

Each recipe writes to its output directory under `$INIT`:

```text
$INIT/<head>_<backbone>_dit_multires/
├── features_*.pt
├── state_dict_dit_blocks_M_enc_blocks_N_post.pt
└── metrics_dit_blocks_M_enc_blocks_N_post.txt
```

`M` is the DiT output and `N` is the reward-encoder output. Each candidate pair
has its own adapter and metric file; the latter contains TRAIN and VAL MSE.
Backbone directory tags are `sd3.5`, `sd3.5_large`, and `flux_dev`.

The script fits the grid; it does not automatically select a pair. Compare
held-out MSE to choose a location. The distillation launcher has
[explicit default layer pairs](STAGE2_VALUE_MODEL_DISTILLATION.md#model-pair-defaults), and the
search produces adapters for those pairs too. If adopting a different pair,
update both `--stitching_layer_location enc_blocks_N_dit_M` and
`--initialization_weight_path` together in the distillation recipe.

Sampling is stochastic, so a new run can produce different fits and layer
choices. These outputs are affine initializations, not trained value models.

### Cache and memory

Existing features and fitted adapters are reused by filename. **Use a fresh
`INIT` directory when changing models, datasets, buckets, extraction count, or
split**; cache filenames do not validate those settings. Feature files are
loaded in full. Plan both CPU RAM and disk for all cached candidate features,
plus reward features, float64 fitting matrices, and model-loading overhead.

Continue with [Stage 2 — Value-model distillation](STAGE2_VALUE_MODEL_DISTILLATION.md).

## Results

The following selected pairs minimize held-out MSE within the documented
candidate grid. All twelve searches use 300 images (200 fit / 100 validation)
and the multi-resolution configuration above. Layer numbers are **one-based
DiT → reward-encoder block outputs**.

| Backbone | Encoder | Candidates | Selected DiT → encoder | Fit MSE | Validation MSE |
|---|---|---:|---:|---:|---:|
| SD3.5 Medium | HPSv2 | 48 | 3 → 1 | 0.018726 | 0.018768 |
| SD3.5 Medium | CLIP | 48 | 3 → 2 | 0.021169 | 0.024964 |
| SD3.5 Medium | Aesthetic | 48 | 3 → 2 | 0.025509 | 0.027853 |
| SD3.5 Medium | DFN-CLIP | 48 | 5 → 2 | 0.038086 | 0.040843 |
| SD3.5 Large | HPSv2 | 76 | 13 → 1 | 0.015711 | 0.020008 |
| SD3.5 Large | CLIP | 76 | 12 → 2 | 0.020625 | 0.023767 |
| SD3.5 Large | Aesthetic | 76 | 12 → 2 | 0.022657 | 0.027757 |
| SD3.5 Large | DFN-CLIP | 76 | 15 → 2 | 0.037036 | 0.040776 |
| FLUX.1-dev | HPSv2 | 112 | 9 → 1 | 0.017050 | 0.020967 |
| FLUX.1-dev | CLIP | 112 | 9 → 2 | 0.021361 | 0.023523 |
| FLUX.1-dev | Aesthetic | 112 | 9 → 2 | 0.024613 | 0.030568 |
| FLUX.1-dev | DFN-CLIP | 112 | 9 → 2 | 0.036072 | 0.041033 |

These layer pairs are the defaults in the distillation launcher. Each row comes
from a stochastic search; fresh sampling can change its MSE and selected pair.
MSE measures the adapter's patch-feature fit and is not a reward-quality score.
Because the target encoders differ, MSE values should be compared within a
model pair's search rather than across reward heads.

See [Stage 2 — Value-model distillation](STAGE2_VALUE_MODEL_DISTILLATION.md) for training these
adapters and the resulting intrinsic benchmark scores.

## Implementation file map

```text
launcher → CLI + image loader → frozen VAE / diffusion backbone / reward encoder
         → paired block features → affine least-squares fits → TRAIN / VAL MSE
         → initialization weight files
```

Read the launcher and CLI first, then the entry point and model loaders,
feature hooks, fitter, and MSE implementation. The table covers the search and
its shared dependencies; distillation-specific files are in the next guide.

| File | Role |
|---|---|
| [find_stitching_layer.py](../find_stitching_layer.py) | Entry point: extract/cache features, split samples, fit each pair, save weights and MSE. |
| [scripts/stage1_find_stitching_layer.sh](../scripts/stage1_find_stitching_layer.sh) | All twelve model-pair commands and output paths. |
| [scripts/paths.sh](../scripts/paths.sh) | Configurable data, model-cache, initialization, and worker settings. |
| [config/stitching_config/base.py](../config/stitching_config/base.py) | Search CLI options, model choices, dataset paths, and convolution spec. |
| [stitchvm/utils/stitching_fit.py](../stitchvm/utils/stitching_fit.py) | Patch extraction/interpolation and joint least-squares fit of weight and bias. |
| [stitchvm/utils/stitching_metrics.py](../stitchvm/utils/stitching_metrics.py) | Held-out and fitting-split spatial MSE. |
| [stitchvm/models/stitching_layer_builder.py](../stitchvm/models/stitching_layer_builder.py) | Convolution specification parser and Conv2d construction. |
| [stitchvm/utils/utils_for_thirdparty.py](../stitchvm/utils/utils_for_thirdparty.py) | Frozen model, VAE, tokenizer, image-processor, and empty-prompt loaders. |
| [stitchvm/models/backbones/flux.py](../stitchvm/models/backbones/flux.py) | FLUX latent packing, guidance, block forwarding, and spatial unpacking. |
| [stitchvm/models/feedforward/clip_head.py](../stitchvm/models/feedforward/clip_head.py) | CLIP feature hooks; also used for the aesthetic encoder. |
| [stitchvm/models/feedforward/hpsv2_head.py](../stitchvm/models/feedforward/hpsv2_head.py) | HPSv2 feature hooks and normalization statistics. |
| [stitchvm/models/feedforward/dfn_clip_head.py](../stitchvm/models/feedforward/dfn_clip_head.py) | DFN feature hooks and normalization statistics. |
| [stitchvm/data/dataset_util.py](../stitchvm/data/dataset_util.py) | Fixed- or variable-resolution dataset assembly and DataLoader. |
| [stitchvm/data/resolution_bucket_sampler.py](../stitchvm/data/resolution_bucket_sampler.py) | Uniform resolution selection per batch and tuple indices across HPDv2 + AVA. |
| [stitchvm/data/hpdv2_datasets.py](../stitchvm/data/hpdv2_datasets.py) | HPDv2 image discovery and paired VAE/reward-image preprocessing. |
| [stitchvm/data/ava_dataset.py](../stitchvm/data/ava_dataset.py) | AVA training split and paired image preprocessing. |
| [scripts/download_weights.py](../scripts/download_weights.py) | Downloads the frozen diffusion and reward-encoder dependencies. |
| [data_processing/prepare_hpdv2_train.py](../data_processing/prepare_hpdv2_train.py) | Downloads/extracts HPDv2 training images. |
| [data_processing/prepare_aesthetic_benchmark.py](../data_processing/prepare_aesthetic_benchmark.py) | Prepares AVA metadata and checks the training image split; filename retained for compatibility. |

The SD3.5 implementation comes from the pinned Diffusers dependency. The fitter
accepts different source grids, and MSE evaluation groups compatible shapes.

### Package entry points

- [stitchvm/__init__.py](../stitchvm/__init__.py)
- [stitchvm/data/__init__.py](../stitchvm/data/__init__.py)
- [stitchvm/models/__init__.py](../stitchvm/models/__init__.py)
- [stitchvm/models/backbones/__init__.py](../stitchvm/models/backbones/__init__.py)
- [stitchvm/models/feedforward/__init__.py](../stitchvm/models/feedforward/__init__.py)
- [stitchvm/utils/__init__.py](../stitchvm/utils/__init__.py)

### Documentation, installation, and licensing

| File | Role |
|---|---|
| [README.md](../README.md) | Project overview, release scope, and relation to the paper. |
| [docs/SETUP.md](SETUP.md) | Shared installation, pretrained models, data, and image preprocessing. |
| [docs/STAGE1_STITCHING_LAYER_SEARCH.md](STAGE1_STITCHING_LAYER_SEARCH.md) | Search setup, configuration, results, outputs, and source map. |
| [docs/STAGE2_VALUE_MODEL_DISTILLATION.md](STAGE2_VALUE_MODEL_DISTILLATION.md) | Distillation setup, configuration, results, outputs, and source map. |
| [requirements.txt](../requirements.txt) | Runtime dependency versions. |
| [pyproject.toml](../pyproject.toml) | Package metadata and discovery. |
| [.gitignore](../.gitignore) | Excludes local data, credentials, caches, weights, and generated files. |
| [LICENSE](../LICENSE) | StitchVM license. |
| [NOTICE](../NOTICE) | Attribution and third-party component notices. |

### Vendored HPSv2 dependency

The HPSv2 loader imports `hpsv2.src.open_clip.create_model`, which uses the
vendored OpenCLIP model library. Model JSON files and the BPE vocabulary are
packaged resources. This dependency is shared by search and distillation.
DFN-CLIP uses the separately installed `open_clip_torch` package.

<details>
<summary>Complete vendored dependency file list</summary>

- [.gitignore](../external_repos/HPSv2/.gitignore)
- [LICENSE](../external_repos/HPSv2/LICENSE)
- [README.md](../external_repos/HPSv2/README.md)
- [hpsv2/__init__.py](../external_repos/HPSv2/hpsv2/__init__.py)
- [hpsv2/src/__init__.py](../external_repos/HPSv2/hpsv2/src/__init__.py)
- [hpsv2/src/open_clip/__init__.py](../external_repos/HPSv2/hpsv2/src/open_clip/__init__.py)
- [hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz](../external_repos/HPSv2/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz)
- [hpsv2/src/open_clip/coca_model.py](../external_repos/HPSv2/hpsv2/src/open_clip/coca_model.py)
- [hpsv2/src/open_clip/constants.py](../external_repos/HPSv2/hpsv2/src/open_clip/constants.py)
- [hpsv2/src/open_clip/factory.py](../external_repos/HPSv2/hpsv2/src/open_clip/factory.py)
- [hpsv2/src/open_clip/hf_configs.py](../external_repos/HPSv2/hpsv2/src/open_clip/hf_configs.py)
- [hpsv2/src/open_clip/hf_model.py](../external_repos/HPSv2/hpsv2/src/open_clip/hf_model.py)
- [hpsv2/src/open_clip/loss.py](../external_repos/HPSv2/hpsv2/src/open_clip/loss.py)
- [hpsv2/src/open_clip/model.py](../external_repos/HPSv2/hpsv2/src/open_clip/model.py)
- [hpsv2/src/open_clip/model_configs/RN101-quickgelu.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/RN101-quickgelu.json)
- [hpsv2/src/open_clip/model_configs/RN101.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/RN101.json)
- [hpsv2/src/open_clip/model_configs/RN50-quickgelu.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/RN50-quickgelu.json)
- [hpsv2/src/open_clip/model_configs/RN50.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/RN50.json)
- [hpsv2/src/open_clip/model_configs/RN50x16.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/RN50x16.json)
- [hpsv2/src/open_clip/model_configs/RN50x4.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/RN50x4.json)
- [hpsv2/src/open_clip/model_configs/RN50x64.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/RN50x64.json)
- [hpsv2/src/open_clip/model_configs/ViT-B-16-plus-240.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-B-16-plus-240.json)
- [hpsv2/src/open_clip/model_configs/ViT-B-16-plus.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-B-16-plus.json)
- [hpsv2/src/open_clip/model_configs/ViT-B-16.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-B-16.json)
- [hpsv2/src/open_clip/model_configs/ViT-B-32-plus-256.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-B-32-plus-256.json)
- [hpsv2/src/open_clip/model_configs/ViT-B-32-quickgelu.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-B-32-quickgelu.json)
- [hpsv2/src/open_clip/model_configs/ViT-B-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-B-32.json)
- [hpsv2/src/open_clip/model_configs/ViT-H-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-H-14.json)
- [hpsv2/src/open_clip/model_configs/ViT-H-16.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-H-16.json)
- [hpsv2/src/open_clip/model_configs/ViT-L-14-280.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-L-14-280.json)
- [hpsv2/src/open_clip/model_configs/ViT-L-14-336.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-L-14-336.json)
- [hpsv2/src/open_clip/model_configs/ViT-L-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-L-14.json)
- [hpsv2/src/open_clip/model_configs/ViT-L-16-320.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-L-16-320.json)
- [hpsv2/src/open_clip/model_configs/ViT-L-16.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-L-16.json)
- [hpsv2/src/open_clip/model_configs/ViT-M-16-alt.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-M-16-alt.json)
- [hpsv2/src/open_clip/model_configs/ViT-M-16.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-M-16.json)
- [hpsv2/src/open_clip/model_configs/ViT-M-32-alt.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-M-32-alt.json)
- [hpsv2/src/open_clip/model_configs/ViT-M-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-M-32.json)
- [hpsv2/src/open_clip/model_configs/ViT-S-16-alt.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-S-16-alt.json)
- [hpsv2/src/open_clip/model_configs/ViT-S-16.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-S-16.json)
- [hpsv2/src/open_clip/model_configs/ViT-S-32-alt.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-S-32-alt.json)
- [hpsv2/src/open_clip/model_configs/ViT-S-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-S-32.json)
- [hpsv2/src/open_clip/model_configs/ViT-bigG-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-bigG-14.json)
- [hpsv2/src/open_clip/model_configs/ViT-e-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-e-14.json)
- [hpsv2/src/open_clip/model_configs/ViT-g-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/ViT-g-14.json)
- [hpsv2/src/open_clip/model_configs/coca_ViT-B-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/coca_ViT-B-32.json)
- [hpsv2/src/open_clip/model_configs/coca_ViT-L-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/coca_ViT-L-14.json)
- [hpsv2/src/open_clip/model_configs/coca_base.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/coca_base.json)
- [hpsv2/src/open_clip/model_configs/coca_roberta-ViT-B-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/coca_roberta-ViT-B-32.json)
- [hpsv2/src/open_clip/model_configs/convnext_base.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_base.json)
- [hpsv2/src/open_clip/model_configs/convnext_base_w.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_base_w.json)
- [hpsv2/src/open_clip/model_configs/convnext_base_w_320.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_base_w_320.json)
- [hpsv2/src/open_clip/model_configs/convnext_large.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_large.json)
- [hpsv2/src/open_clip/model_configs/convnext_large_d.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_large_d.json)
- [hpsv2/src/open_clip/model_configs/convnext_large_d_320.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_large_d_320.json)
- [hpsv2/src/open_clip/model_configs/convnext_small.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_small.json)
- [hpsv2/src/open_clip/model_configs/convnext_tiny.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_tiny.json)
- [hpsv2/src/open_clip/model_configs/convnext_xlarge.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_xlarge.json)
- [hpsv2/src/open_clip/model_configs/convnext_xxlarge.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_xxlarge.json)
- [hpsv2/src/open_clip/model_configs/convnext_xxlarge_320.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/convnext_xxlarge_320.json)
- [hpsv2/src/open_clip/model_configs/mt5-base-ViT-B-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/mt5-base-ViT-B-32.json)
- [hpsv2/src/open_clip/model_configs/mt5-xl-ViT-H-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/mt5-xl-ViT-H-14.json)
- [hpsv2/src/open_clip/model_configs/roberta-ViT-B-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/roberta-ViT-B-32.json)
- [hpsv2/src/open_clip/model_configs/swin_base_patch4_window7_224.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/swin_base_patch4_window7_224.json)
- [hpsv2/src/open_clip/model_configs/vit_medium_patch16_gap_256.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/vit_medium_patch16_gap_256.json)
- [hpsv2/src/open_clip/model_configs/vit_relpos_medium_patch16_cls_224.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/vit_relpos_medium_patch16_cls_224.json)
- [hpsv2/src/open_clip/model_configs/xlm-roberta-base-ViT-B-32.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/xlm-roberta-base-ViT-B-32.json)
- [hpsv2/src/open_clip/model_configs/xlm-roberta-large-ViT-H-14.json](../external_repos/HPSv2/hpsv2/src/open_clip/model_configs/xlm-roberta-large-ViT-H-14.json)
- [hpsv2/src/open_clip/modified_resnet.py](../external_repos/HPSv2/hpsv2/src/open_clip/modified_resnet.py)
- [hpsv2/src/open_clip/openai.py](../external_repos/HPSv2/hpsv2/src/open_clip/openai.py)
- [hpsv2/src/open_clip/pretrained.py](../external_repos/HPSv2/hpsv2/src/open_clip/pretrained.py)
- [hpsv2/src/open_clip/push_to_hf_hub.py](../external_repos/HPSv2/hpsv2/src/open_clip/push_to_hf_hub.py)
- [hpsv2/src/open_clip/timm_model.py](../external_repos/HPSv2/hpsv2/src/open_clip/timm_model.py)
- [hpsv2/src/open_clip/tokenizer.py](../external_repos/HPSv2/hpsv2/src/open_clip/tokenizer.py)
- [hpsv2/src/open_clip/transform.py](../external_repos/HPSv2/hpsv2/src/open_clip/transform.py)
- [hpsv2/src/open_clip/transformer.py](../external_repos/HPSv2/hpsv2/src/open_clip/transformer.py)
- [hpsv2/src/open_clip/utils.py](../external_repos/HPSv2/hpsv2/src/open_clip/utils.py)
- [hpsv2/src/open_clip/version.py](../external_repos/HPSv2/hpsv2/src/open_clip/version.py)
- [setup.py](../external_repos/HPSv2/setup.py)

</details>
