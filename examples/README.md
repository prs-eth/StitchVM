# Inference examples

Score your own images with a StitchVM checkpoint at a chosen latent noise level.
The same entry point covers all 16 supported backbone–reward combinations.

## Setup

Follow [Setup](../docs/SETUP.md) first. HPSv3 and UnifiedReward also require the
additional dependencies in their [HPSv3](../docs/EXTENSION_HPSV3.md#setup) and
[UnifiedReward](../docs/EXTENSION_UNIFIEDREWARD_QWEN35.md#setup) guides.
Run commands from the repository root, on a CUDA GPU with enough memory for the
selected backbone and reward model. The language models remain loaded during
HPSv3 and UnifiedReward inference; the 27B judge requires substantially more
memory than the smaller models.

Accept the SD3.5 or FLUX model license on Hugging Face and run `hf auth login`.
Download the upstream dependencies you need, for example:

```bash
python scripts/download_weights.py stable_diffusion_3 clip
# Other reward choices: hpsv2, aesthetic, dfn_clip, hpsv3,
# unifiedreward_qwen35_4b, unifiedreward_qwen35_9b, unifiedreward_qwen35_27b
python scripts/download_weights.py --list
```

Downloaded reward files default to `reward_ckpts/`. Set `STITCHVM_REWARD_CKPTS`
when using a different location. The example downloads the selected StitchVM
EMA checkpoint from [HJGO/stitchvm](https://huggingface.co/HJGO/stitchvm), or you
can supply `--checkpoint /path/to/model_trainables_ema.pt`.

## Choose a reward model

| `--model` | Supported `--backbone` | Inputs besides the image | Output |
|---|---|---|---|
| `hpsv2` | `sd3.5`, `sd3.5_large`, `flux_dev` | `--prompt` | Image–text cosine similarity |
| `clip` | `sd3.5`, `sd3.5_large`, `flux_dev` | `--prompt` | Image–text cosine similarity |
| `aesthetic` | `sd3.5`, `sd3.5_large`, `flux_dev` | None | Aesthetic MLP score |
| `dfn_clip` | `sd3.5`, `sd3.5_large`, `flux_dev` | `--prompt` | Image–text cosine similarity |
| `hpsv3` | `sd3.5` | `--prompt` | HPSv3 reward (μ) |
| `unifiedreward_4b` | `sd3.5` | `--question`, `--answer1`, `--answer2` | Generated explanation and preference verdict |
| `unifiedreward_9b` | `sd3.5` | Same as above | Same as above |
| `unifiedreward_27b` | `sd3.5` | Same as above | Same as above |

`sd3.5` means SD3.5 Medium. List all combinations without loading weights:

```bash
python examples/inference.py --list
```

### CLIP, HPSv2 and DFN-CLIP

Replace `photo.jpg` with your image. Change `--model` to `hpsv2` or `dfn_clip`
to use the other text–image reward encoders. Use `--backbone sd3.5_large` or
`--backbone flux_dev` to select another diffusion backbone.

```bash
python examples/inference.py --model clip --backbone sd3.5 \
  --image photo.jpg --prompt "A red bicycle leaning against a brick wall" \
  --sigma 0.25 --seed 0
```

Each image produces one JSON line containing `score`, `score_type`, model,
backbone, sigma, and seed. Similarities are normalized dot products, without
multiplication by 100 or CLIP's learned logit scale. Larger scores indicate
better alignment with the given prompt; they are not probabilities.

### Aesthetic

```bash
python examples/inference.py --model aesthetic --backbone sd3.5 \
  --image photo.jpg --sigma 0.25 --seed 0
```

Aesthetic uses the frozen MLP on the normalized stitched image embedding and
does not use a text prompt.

### HPSv3

```bash
python examples/inference.py --model hpsv3 \
  --image photo.jpg --prompt "A red bicycle leaning against a brick wall" \
  --sigma 0.25 --seed 0
```

The trained vision branch supplies tokens to the original HPSv3 language model
and reward head. `--reward-checkpoint /path/to/HPSv3.safetensors` optionally
selects a local copy of the pinned upstream HPSv3 weights.

### UnifiedReward

```bash
python examples/inference.py --model unifiedreward_4b \
  --image photo.jpg --question "What is leaning against the wall?" \
  --answer1 "A red bicycle." --answer2 "A blue car." \
  --sigma 0.25 --seed 0
```

Change the model to `unifiedreward_9b` or `unifiedreward_27b` for another size.
`--reward-checkpoint /path/to/UnifiedReward-2.0-qwen35-4b` overrides the default
upstream reward directory. It must match the selected StitchVM weights.

This interface judges two supplied answers. It returns the generated text,
the official phrase-based verdict, and the explicit-preference parse. An
unrecognized or ambiguous verdict remains unparsed; it is not forced into a
choice. `--max-new-tokens` defaults to 4096; a shorter limit can truncate the
verdict. See the [evaluation protocol](../docs/EXTENSION_UNIFIEDREWARD_QWEN35.md)
for the fixed randomized answer order and benchmark aggregation.

## Noise and image size

All examples encode the image with the diffusion VAE and use
`z_sigma = (1 - sigma) * z + sigma * epsilon`, with `epsilon ~ N(0, I)`.
`sigma=0` uses the clean latent; `sigma=1` uses pure noise. The diffusion backbone
receives the empty prompt, as in training. The user prompt is used by the reward
model. The same seed repeats the noise draw for the same image shape and device.

CLIP, HPSv2, Aesthetic and DFN-CLIP default to a 1024×1024 VAE resize and their
trained reward-head grids. `--vae-input-size 512` or `768` selects another
square resolution. HPSv3 also defaults to a 1024×1024 VAE input but uses the
original image's HPSv3 token grid. UnifiedReward uses its processor-native
image grid; it does not accept `--vae-input-size`.

Pass several images to score them sequentially with the same prompt:

```bash
python examples/inference.py --model clip --image first.jpg second.jpg \
  --prompt "A red bicycle" --sigma 0.1
```

To compare noise levels, repeat the command with `--sigma 0`, `0.1`, `0.25`,
`0.5`, `0.75`, or `0.9`. Scores from different reward models have different
scales and should not be compared directly.

These examples score images. They do not run an image-generation or
inference-time alignment algorithm.
