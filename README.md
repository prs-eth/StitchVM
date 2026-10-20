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

**StitchVM** turns pretrained pixel reward models (CLIP, HPSv2, Aesthetic
Predictor, and DFN-CLIP) into **value models that score noisy diffusion latents
directly**, at small compute cost. The resulting value models can be used in
diffusion-alignment methods such as DPS, FK steering, DRaFT, and DiffusionNFT.

See the **[project page](https://gohyojun15.github.io/StitchVM/)** for the full
story, figures, and results.

---

## Code and checkpoint availability

The code supports stitching-layer search and value-model distillation for all
12 combinations of **SD3.5 Medium, SD3.5 Large, and FLUX.1-dev** with
**HPSv2, CLIP, Aesthetic, and DFN-CLIP** reward encoders.

**TODO: StitchVM checkpoints have not been released.** Precomputed stitching
initializations are also not included. The guides below explain how to generate
initializations and train models locally. The download helper fetches the
original pretrained diffusion and reward models needed by these workflows.

## TODO

- [x] Release pretrained StitchVM checkpoints.
- [ ] Add support for additional diffusion backbones and reward models.
- [ ] Release inference-time alignment code.

Want to use StitchVM with a particular diffusion backbone and reward model?
Submit a [model integration request](https://docs.google.com/forms/d/e/1FAIpQLSebA4Ac9-AQuUCbN_wkYNt6Qc0ZImG06DvFjyw1_vBsC28Wjw/viewform)
with your model combination, use case, and where you would like help.


## Training recipe and relation to the paper

**The training recipe in this repository differs from the paper.** We introduced
multi-resolution training to improve support across image resolutions. The
current recipes use 512×512, 768×768, 1024×1024, 768×1024, and 1024×768 VAE inputs;
DFN-CLIP uses the three square resolutions. Reward encoders retain their native
input sizes, and stitching-layer search uses the same resolution buckets as
training.

The documentation describes the current repository configuration. It should not
be treated as an exact reproduction of the paper's training setup or reported
results.

## Setup

Follow [shared setup](docs/SETUP.md) to install the environment, download the
pretrained diffusion/reward dependencies, and prepare **HPDv2 + AVA**. It also
explains the `DATA`, `INIT`, and `OUT` paths used by the commands below.

The stage numbers are **repository workflow labels**, distinct from the paper's
terminology. Each guide combines setup, launch commands, detailed configuration,
results, and an implementation file map.

## How to run

### Stage 1 — Stitching-layer search

Extract paired features from the frozen diffusion backbone and reward encoder,
then fit an affine adapter for each candidate pair. Each search uses **300
images: 200 for fitting and 100 for validation**, with the first half of DiT
blocks and the first four reward-encoder blocks as candidates. VAE resolutions
match the training buckets; reward inputs keep their native sizes.

```bash
# List the twelve recipes, then run CLIP with SD3.5 Medium.
bash scripts/stage1_find_stitching_layer.sh
bash scripts/stage1_find_stitching_layer.sh stage1_clip
```

See the [full search guide](docs/STAGE1_STITCHING_LAYER_SEARCH.md) for all twelve recipes,
configuration details, output files, and how to choose a layer pair.

### Stage 2 — Value-model distillation

Initialize from the fitted adapter, then train the adapter and reward-encoder
suffix for **five epochs**, keeping the VAE, diffusion backbone, and reward
teacher frozen. The recipes use multi-resolution inputs, **AdamW**, and
**EMA decay 0.999**.

Base learning rates are **5e-5** for SD3.5 Medium and FLUX, and **1e-5** for
SD3.5 Large, with adapter multiplier 1. Global batch is **256** for SD3.5
HPSv2/CLIP/Aesthetic and **128** for FLUX and all DFN-CLIP pairs; accumulation
preserves these targets when changing the per-device micro-batch.

```bash
# Preview the resolved settings, then train the matching pair.
DRY_RUN=1 bash scripts/stage2_train_stitched_model.sh stage2_clip
bash scripts/stage2_train_stitched_model.sh stage2_clip
```

See the [full distillation guide](docs/STAGE2_VALUE_MODEL_DISTILLATION.md) for pair-specific layers,
learning rates, batch sizes, losses, memory settings, and checkpoint outputs.

## Results

### Stage 1 — Stitching-layer search results

Selected **DiT → reward-encoder block outputs** (one-based):

| Backbone | HPSv2 | CLIP | Aesthetic | DFN-CLIP |
|---|---:|---:|---:|---:|
| SD3.5 Medium | 3 → 1 | 3 → 2 | 3 → 2 | 5 → 2 |
| SD3.5 Large | 13 → 1 | 12 → 2 | 12 → 2 | 15 → 2 |
| FLUX.1-dev | 9 → 1 | 9 → 2 | 9 → 2 | 9 → 2 |

The [full search results](docs/STAGE1_STITCHING_LAYER_SEARCH.md#results) include candidate counts,
fit MSE, and validation MSE for all twelve pairs. Fresh stochastic searches can
produce different values.

### Stage 2 — Value-model distillation results

Reported intrinsic evaluation after five epochs, using **epoch-5 EMA weights**
and **1024×1024** evaluation inputs. Each cell is the best primary score over
the reported noise sweep σ ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9}.

| Backbone | HPSv2 preference accuracy | CLIP Image R@1 | Aesthetic SRCC | DFN-CLIP Image R@1 |
|---|---:|---:|---:|---:|
| SD3.5 Medium | 81.70% | 39.14% | 0.6306 | 54.21% |
| SD3.5 Large | 82.01% | 39.16% | 0.6315 | 54.29% |
| FLUX.1-dev | 82.15% | 39.29% | 0.6335 | 54.37% |

Benchmarks: HPDv2 test for HPSv2, MSCOCO 5k for retrieval, and AVA 20k for
Aesthetic. The [full benchmark results](docs/STAGE2_VALUE_MODEL_DISTILLATION.md#results) include
peak noise levels, unstitched baselines, and all supplied metric breakdowns.
The retrieval breakdown covers selected noise levels. **Checkpoints are not yet
released.** Standalone intrinsic-evaluation and diffusion-alignment launchers
are not included in this code release.

### Pretrained checkpoints

Pretrained StitchVM EMA checkpoints are available on
[Hugging Face](https://huggingface.co/HJGO/stitchvm) for all 12 supported
backbone–reward combinations.

Layer pairs follow the training configurations and denote
**DiT → reward-encoder block outputs**, indexed from 1.

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
