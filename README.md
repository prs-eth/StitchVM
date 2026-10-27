<div align="center">

# Stitched Value Model for Diffusion Alignment

**NeurIPS 2026**

[Hyojun Go](https://gohyojun15.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Hyungjin Chung](https://hyungjin-chung.github.io/) &nbsp;·&nbsp;
[Prune Truong](https://prunetruong.com/)<sup>2</sup> &nbsp;·&nbsp;
[Goutam Bhat](https://goutamgmb.github.io/)<sup>2</sup> &nbsp;·&nbsp;
[Li Mi](https://limirs.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Zhaochong An](https://zhaochongan.github.io/)<sup>3</sup>
<br>
[Zixiang Zhao](https://zhaozixiang1228.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Dominik Narnhofer](https://baug.ethz.ch/departement/personen/mitarbeiter/personen-detail.MzM5ODU5.TGlzdC82NzksLTU1NTc1NDEwMQ==.html)<sup>1</sup>
<br>
[Serge Belongie](https://sergebelongie.github.io/)<sup>3</sup> &nbsp;·&nbsp;
[Federico Tombari](https://federicotombari.github.io/)<sup>2</sup> &nbsp;·&nbsp;
[Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)<sup>1</sup>

<sup>1</sup>ETH Zürich &nbsp;·&nbsp;
<sup>2</sup>Google &nbsp;·&nbsp;
<sup>3</sup>University of Copenhagen

[**Project page**](https://gohyojun15.github.io/StitchVM/) &nbsp;|&nbsp;
[**Paper (PDF)**](https://gohyojun15.github.io/StitchVM/static/stitchvm.pdf) &nbsp;|&nbsp;
[**arXiv**](https://arxiv.org/abs/2605.19804)

</div>

---

<p align="center">
  <img src="https://gohyojun15.github.io/StitchVM/static/images/fig_overview-1.png" alt="StitchVM method overview" width="92%">
</p>

**StitchVM** turns pretrained reward-model vision encoders into **value models
that score noisy diffusion latents directly**. It supports CLIP, HPSv2,
Aesthetic, DFN-CLIP, HPSv3, and UnifiedReward-2.0 Qwen3.5. Vision-language models
retain their original language branches for reward scoring or answer comparison.

See the **[project page](https://gohyojun15.github.io/StitchVM/)** for the method,
figures, and diffusion-alignment results.

[Setup](#setup) · [Inference](#inference) · [Checkpoints](#pretrained-checkpoints) ·
[Training](#training) · [Results](#results)

## News

- Added **HPSv3** and **UnifiedReward-2.0 Qwen3.5 (4B, 9B, 27B)** support with
  SD3.5 Medium, including training, inference, evaluation, and
  [four pretrained EMA checkpoints](#vision-language-reward-models).
- A [shared inference example](examples/README.md) now covers **all 16 supported
  combinations**, with image inputs and a configurable noise level.

## Setup

Follow [the setup guide](docs/SETUP.md#install) to install the environment and
[download the upstream models](docs/SETUP.md#download-pretrained-dependencies).
HPSv3 and UnifiedReward have additional instructions in their
[HPSv3](docs/EXTENSION_HPSV3.md#setup) and
[UnifiedReward](docs/EXTENSION_UNIFIEDREWARD_QWEN35.md#setup) guides.

Inference requires model weights and your input image. **HPDv2 + AVA are needed
only for stitching-layer search and training**, not for scoring your own images.

## Inference

Run from the repository root after setup. Replace `photo.jpg` with your image:

```bash
python examples/inference.py --model clip --backbone sd3.5 \
  --image photo.jpg --prompt "A red bicycle leaning against a brick wall" --sigma 0.25
```

The command downloads the matching StitchVM EMA checkpoint automatically.
Use `--checkpoint /path/to/model_trainables_ema.pt` to load a local copy, or
`python examples/inference.py --list` to list supported combinations.

| Reward model | Input | Output |
|---|---|---|
| CLIP, HPSv2, DFN-CLIP | Image + prompt | Image–text cosine similarity |
| Aesthetic | Image | Aesthetic score |
| HPSv3 | Image + prompt | HPSv3 reward |
| UnifiedReward-2.0 Qwen3.5 | Image + question + two candidate answers | Explanation and preference verdict |

See [all inference examples](examples/README.md) for model-specific commands,
noise levels, image sizes, and GPU requirements. These examples score images;
diffusion-alignment methods such as DPS and FK steering are not included yet.

## Pretrained checkpoints

**All 16 supported combinations** have epoch-5 EMA checkpoints on
[Hugging Face](https://huggingface.co/HJGO/stitchvm). Layer pairs denote
**DiT → reward-encoder block outputs**, indexed from 1. Upstream diffusion and
reward weights are loaded separately and retain their respective licenses.

### CLIP-based reward encoders

| Diffusion backbone | Reward model | DiT → Encoder | EMA checkpoint |
|---|---|:---:|:---:|
| SD3.5 Medium | HPSv2 | 3 → 1 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/hpsv2_sd3.5_medium/model_trainables_ema.pt) |
| SD3.5 Medium | CLIP | 3 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/clip_sd3.5_medium/model_trainables_ema.pt) |
| SD3.5 Medium | Aesthetic | 3 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_aesthetic_sd3/checkpoints/epoch_0005/model_trainables_ema.pt) |
| SD3.5 Medium | DFN-CLIP | 5 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_dfn_clip_sd3/checkpoints/epoch_0005/model_trainables_ema.pt) |
| SD3.5 Large | HPSv2 | 13 → 1 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_hpsv2_sd3_large/checkpoints/epoch_0005/model_trainables_ema.pt) |
| SD3.5 Large | CLIP | 12 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_clip_sd3_large/checkpoints/epoch_0005/model_trainables_ema.pt) |
| SD3.5 Large | Aesthetic | 12 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_aesthetic_sd3_large/checkpoints/epoch_0005/model_trainables_ema.pt) |
| SD3.5 Large | DFN-CLIP | 15 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_dfn_clip_sd3_large/checkpoints/epoch_0005/model_trainables_ema.pt) |
| FLUX.1-dev | HPSv2 | 9 → 1 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_hpsv2_flux/checkpoints/epoch_0005/model_trainables_ema.pt) |
| FLUX.1-dev | CLIP | 9 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_clip_flux/checkpoints/epoch_0005/model_trainables_ema.pt) |
| FLUX.1-dev | Aesthetic | 9 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_aesthetic_flux/checkpoints/epoch_0005/model_trainables_ema.pt) |
| FLUX.1-dev | DFN-CLIP | 9 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/stitchvm_dfn_clip_flux/checkpoints/epoch_0005/model_trainables_ema.pt) |

### Vision-language reward models

These four combinations use **SD3.5 Medium**. Training matches post-merger
image tokens with MSE; the language model is used only for inference and evaluation.

| Reward model | DiT → encoder | Epoch-5 EMA | Guide |
|---|:---:|:---:|---|
| HPSv3 (Qwen2-VL-7B) | 5 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/hpsv3_sd3.5_medium/model_trainables_ema.pt) | [Setup, training, inference and results](docs/EXTENSION_HPSV3.md) |
| UnifiedReward-2.0 Qwen3.5 4B | 4 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/unifiedreward_qwen35_4b_sd3.5_medium/model_trainables_ema.pt) | [Guide](docs/EXTENSION_UNIFIEDREWARD_QWEN35.md) |
| UnifiedReward-2.0 Qwen3.5 9B | 3 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/unifiedreward_qwen35_9b_sd3.5_medium/model_trainables_ema.pt) | [Guide](docs/EXTENSION_UNIFIEDREWARD_QWEN35.md) |
| UnifiedReward-2.0 Qwen3.5 27B | 3 → 2 | [Download](https://huggingface.co/HJGO/stitchvm/resolve/main/unifiedreward_qwen35_27b_sd3.5_medium/model_trainables_ema.pt) | [Guide](docs/EXTENSION_UNIFIEDREWARD_QWEN35.md) |

The model guides include exact configurations, inference APIs, evaluation
commands, and results. Checkpoint folders contain model metadata and checksums.

## Training

The stage numbers are **repository workflow labels**, distinct from the paper's
terminology. Prepare [HPDv2 + AVA and the data paths](docs/SETUP.md), then follow
the search and distillation guides.

### Training recipe and relation to the paper

**The training recipe in this repository differs from the paper.** We introduced
multi-resolution training to improve support across image resolutions. Search
and training use 512×512, 768×768, 1024×1024, 768×1024, and 1024×768 VAE inputs;
DFN-CLIP uses the three square sizes. The model guides specify each reward
encoder's input size and token grid.

The configurations and results below describe this repository's recipe and
should not be treated as an exact reproduction of the paper's training setup.

### Stage 1 — Stitching-layer search

Extract paired features from frozen diffusion and reward encoders, then fit an
affine adapter for each candidate pair. Searches use **300 images: 200 for
fitting and 100 for validation**, the first half of DiT blocks, and the first
four reward-encoder blocks. Search and training use the same VAE resolution buckets.

```bash
# List available recipes, then search CLIP with SD3.5 Medium.
bash scripts/stage1_find_stitching_layer.sh
bash scripts/stage1_find_stitching_layer.sh stage1_clip
```

The [search guide](docs/STAGE1_STITCHING_LAYER_SEARCH.md) covers the original
12 combinations. The [HPSv3](docs/EXTENSION_HPSV3.md#configuration) and
[UnifiedReward](docs/EXTENSION_UNIFIEDREWARD_QWEN35.md#configuration) guides
cover the four vision-language combinations. Each includes settings, results,
and an implementation file map.

### Stage 2 — Value-model distillation

Initialize from the fitted adapter and train the adapter and reward-encoder
suffix for **five epochs**, with **AdamW** and **EMA decay 0.999**. The VAE,
diffusion backbone, and teacher remain frozen.

Base learning rates are **5e-5** for SD3.5 Medium and FLUX, and **1e-5** for
SD3.5 Large, with adapter multiplier 1. Global batch is **256** for SD3.5
HPSv2/CLIP/Aesthetic and the four vision-language combinations, and **128** for
FLUX and all DFN-CLIP pairs. Gradient accumulation preserves these targets when
the per-device micro-batch changes.

```bash
# Inspect the resolved settings, then train the matching pair.
DRY_RUN=1 bash scripts/stage2_train_stitched_model.sh stage2_clip
bash scripts/stage2_train_stitched_model.sh stage2_clip
```

The [distillation guide](docs/STAGE2_VALUE_MODEL_DISTILLATION.md) and the
[HPSv3](docs/EXTENSION_HPSV3.md#configuration) /
[UnifiedReward](docs/EXTENSION_UNIFIEDREWARD_QWEN35.md#configuration) guides
specify layer selections, losses, batch sizes, memory settings, and output paths.

## Results

### Stage 1 — Stitching-layer search results

Selected **DiT → reward-encoder block outputs** (one-based):

| Backbone | HPSv2 | CLIP | Aesthetic | DFN-CLIP |
|---|---:|---:|---:|---:|
| SD3.5 Medium | 3 → 1 | 3 → 2 | 3 → 2 | 5 → 2 |
| SD3.5 Large | 13 → 1 | 12 → 2 | 12 → 2 | 15 → 2 |
| FLUX.1-dev | 9 → 1 | 9 → 2 | 9 → 2 | 9 → 2 |

The four vision-language layer pairs appear in the
[checkpoint table](#vision-language-reward-models). The
[full search results](docs/STAGE1_STITCHING_LAYER_SEARCH.md#results) and model
guides include candidate counts, fit MSE, and validation MSE. Fresh stochastic
searches can produce different values.

### Stage 2 — Value-model distillation results

For the original 12 combinations, reported intrinsic evaluations use
**epoch-5 EMA weights** and **1024×1024 VAE inputs**. Each cell is the best
primary score over σ ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9}.

| Backbone | HPSv2 preference accuracy | CLIP Image R@1 | Aesthetic SRCC | DFN-CLIP Image R@1 |
|---|---:|---:|---:|---:|
| SD3.5 Medium | 81.70% | 39.14% | 0.6306 | 54.21% |
| SD3.5 Large | 82.01% | 39.16% | 0.6315 | 54.29% |
| FLUX.1-dev | 82.15% | 39.29% | 0.6335 | 54.37% |

Benchmarks: HPDv2 test for HPSv2, MSCOCO 5k for retrieval, and AVA 20k for
Aesthetic. [Detailed results](docs/STAGE2_VALUE_MODEL_DISTILLATION.md#results)
include peak noise levels, unstitched baselines, and the supplied breakdowns.
Standalone benchmark launchers for these 12 combinations are not included.

### Vision-language reward results

Reported epoch-5 EMA results with **SD3.5 Medium**. Each row shows the best
stitched score over the evaluated noise levels, with the original model scored
under the same protocol. Accuracy values are percentages.

| Reward model | Benchmark / metric | Original | Best stitched | σ |
|---|---|---:|---:|---:|
| HPSv3 | HPDv2 inversion-score accuracy | 85.45 | 85.00 | 0.1 |
| HPSv3 | ImageReward WebP pairwise accuracy | 67.77 | 68.20 | 0.1 |
| UnifiedReward Qwen3.5 4B | VL-RewardBench accuracy, official parser | 75.30 | 73.14 | 0.1 |
| UnifiedReward Qwen3.5 9B | VL-RewardBench accuracy, official parser | 84.36 | 83.56 | 0.25 |
| UnifiedReward Qwen3.5 27B | VL-RewardBench accuracy, official parser | 80.91 | 80.99 | 0.25 |

HPSv3 uses 1024×1024 VAE inputs and the native HPSv3 head grid; its sweep is
σ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}. UnifiedReward uses processor-native image sizes
and all six noise levels above. Its 1,247-item benchmark uses the same fixed,
randomized answer order for both models, with all items included in the accuracy
denominator. An additional explicit-preference parser is applied equally to
original and stitched outputs.
See the [HPSv3 results](docs/EXTENSION_HPSV3.md#results) and
[UnifiedReward results](docs/UNIFIEDREWARD_QWEN35_RESULTS.md) for all noise
levels, both parsers, and evaluation details. Evaluation scripts are included
for these models.

## TODO

- [x] Release pretrained StitchVM checkpoints for the original 12 combinations.
- [x] Support HPSv3 and UnifiedReward-2.0 Qwen3.5, with pretrained checkpoints.
- [x] Provide image-scoring examples for all supported models.
- [ ] Add support for additional diffusion backbones and reward models.
- [ ] Release inference-time alignment code.

Want to use StitchVM with a particular diffusion backbone and reward model?
Submit a [model integration request](https://docs.google.com/forms/d/e/1FAIpQLSebA4Ac9-AQuUCbN_wkYNt6Qc0ZImG06DvFjyw1_vBsC28Wjw/viewform)
with your model combination, use case, and where you would like help.

## Citation

```bibtex
@inproceedings{go2026stitchvm,
  title     = {Stitched Value Model for Diffusion Alignment},
  author    = {Go, Hyojun and Chung, Hyungjin and Truong, Prune and Bhat, Goutam and
               Mi, Li and An, Zhaochong and Zhao, Zixiang and Narnhofer, Dominik and
               Belongie, Serge and Tombari, Federico and Schindler, Konrad},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Contact

For questions about the method or paper, please open a GitHub issue or contact
[Hyojun Go](mailto:gohyojun15@gmail.com).

Licensed under Apache-2.0 (see [LICENSE](LICENSE), [NOTICE](NOTICE)).
