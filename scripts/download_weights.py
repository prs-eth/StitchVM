#!/usr/bin/env python
"""Pre-download the backbone and reward models for Stage 1 feature extraction.

Run this once with internet access, pointing HF_HOME at your model cache:

    export HF_HOME=/path/to/big/disk/hf_cache
    python scripts/download_weights.py --list             # see what's available
    python scripts/download_weights.py --all              # everything
    python scripts/download_weights.py stable_diffusion_3 hpsv2

For subsequent offline use:

    export HF_HOME=/path/to/big/disk/hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

Everything loads from the cache; nothing touches the network.

Notes
- Only dependency: `huggingface_hub` (pip install huggingface_hub).
- SD 3.5 and FLUX are GATED — accept each license on its Hub page and run
  `hf auth login` once, or the download fails with 401/403.
- hpsv2 also places files under `<repo>/reward_ckpts/`
  (see the layout in README) — run from the repo, or set
  STITCHVM_REWARD_CKPTS to the directory you want.
"""

import argparse
import os
import shutil
import sys
import urllib.request

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_CKPTS = os.environ.get(
    "STITCHVM_REWARD_CKPTS", os.path.join(REPO_ROOT, "reward_ckpts")
)

# Diffusion backbones ship in TWO layouts in the same repo: the diffusers
# component tree (model_index.json + transformer/ + vae/ + text_encoder*/ +
# tokenizer*/ + scheduler/) that this codebase loads via from_pretrained, and
# large single-file exports (ComfyUI `sd3.5_*.safetensors` / `flux1-dev.*`,
# the BFL `ae.safetensors`) that it does NOT use. We fetch ONLY the diffusers
# tree via allow_patterns — an explicit allow-list guarantees the vae/ folder
# comes down (a component the model can't run without) and drops the redundant
# single-file exports. `*` matches across `/`, so `vae/*` covers the whole
# subfolder.
_SD3_ALLOW = [
    "model_index.json", "scheduler/*", "transformer/*", "vae/*",
    "text_encoder/*", "text_encoder_2/*", "text_encoder_3/*",
    "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
]
_FLUX_ALLOW = [
    "model_index.json", "scheduler/*", "transformer/*", "vae/*",
    "text_encoder/*", "text_encoder_2/*",
    "tokenizer/*", "tokenizer_2/*",
]
# The text encoders ship both fp32 and fp16 weights; the pipelines load the
# fp32 files (no variant= is passed), so skip the fp16 duplicates.
_FP16_IGNORE = ["*.fp16.safetensors", "*.fp16-*.safetensors", "*.index.fp16.json"]


def _snap(repo_id, **kw):
    print(f"  ↓ {repo_id}")
    path = snapshot_download(repo_id=repo_id, **kw)
    print(f"    -> {path}")


def _file(repo_id, filename, dest_dir):
    print(f"  ↓ {repo_id}/{filename}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(filename))
    if not os.path.exists(dest):
        shutil.copy(cached, dest)
    print(f"    -> {dest}")


# --------------------------------------------------------------------------- #
# Diffusion backbones (gated — accept the license + `hf auth login`)
# --------------------------------------------------------------------------- #

def stable_diffusion_3():
    """SD 3.5 Medium — the main paper backbone (transformer + vae + text encoders)."""
    _snap("stabilityai/stable-diffusion-3.5-medium",
          allow_patterns=_SD3_ALLOW, ignore_patterns=_FP16_IGNORE)


def stable_diffusion_3_large():
    """SD 3.5 Large (transformer + vae + text encoders)."""
    _snap("stabilityai/stable-diffusion-3.5-large",
          allow_patterns=_SD3_ALLOW, ignore_patterns=_FP16_IGNORE)


def flux_dev():
    """FLUX.1-dev (transformer + vae + text encoders)."""
    _snap("black-forest-labs/FLUX.1-dev",
          allow_patterns=_FLUX_ALLOW, ignore_patterns=_FP16_IGNORE)


# --------------------------------------------------------------------------- #
# Reward encoders
# --------------------------------------------------------------------------- #

def clip():
    """OpenAI CLIP-L/14 (224 + 336) — CLIP head and aesthetic backbone."""
    _snap("openai/clip-vit-large-patch14")
    _snap("openai/clip-vit-large-patch14-336")


def dfn_clip():
    """Apple DFN5B ViT-H/14 (378) — DFN-CLIP head."""
    _snap("apple/DFN5B-CLIP-ViT-H-14-378")



def hpsv2():
    """HPSv2.1 weights + open_clip ViT-H-14 backbone -> reward_ckpts/hpsv2/ (~6 GB)."""
    dest = os.path.join(REWARD_CKPTS, "hpsv2")
    _file("xswu/HPSv2", "HPS_v2.1_compressed.pt", dest)
    _file("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "open_clip_pytorch_model.bin", dest)


# Extension reward models, pinned to the revisions used for the documented runs.
_HPSV3 = ("MizzenAI/HPSv3", "4f81e3e09edd82fe3c5f636444c721b592a735ca")
_QWEN2VL_7B = ("Qwen/Qwen2-VL-7B-Instruct", "eed13092ef92e448dd6875b2a00151bd3f7db0ac")
_UNIFIEDREWARD_QWEN35 = {
    "4b": "a1cb85ca3dd2c65ff561cc1d8fd66e6830c5ab30",
    "9b": "f01548b009741e12ff9817ed91dba94701ed9579",
    "27b": "3a5c836f485047a96f45754e08c981ca9e41fa60",
}


def hpsv3():
    """HPSv3 checkpoint + Qwen2-VL-7B-Instruct base (HF cache, pinned; ~32 GB)."""
    repo, revision = _HPSV3
    print(f"  ↓ {repo}@{revision[:8]}/HPSv3.safetensors")
    print(f"    -> {hf_hub_download(repo_id=repo, filename='HPSv3.safetensors', revision=revision)}")
    _snap(_QWEN2VL_7B[0], revision=_QWEN2VL_7B[1])


def _unifiedreward_qwen35(size):
    repo = f"CodeGoat24/UnifiedReward-2.0-qwen35-{size}"
    revision = _UNIFIEDREWARD_QWEN35[size]
    dest = os.path.join(REWARD_CKPTS, f"UnifiedReward-2.0-qwen35-{size}")
    _snap(repo, revision=revision, local_dir=dest)
    # Read by the Qwen3.5 identity guards (search sidecar, checkpoint reward_meta).
    with open(os.path.join(dest, "PINNED_REVISION"), "w") as f:
        f.write(f"{repo}@{revision}\n")


def unifiedreward_qwen35_4b():
    """UnifiedReward-2.0-qwen35-4b (pinned, ~10 GB) -> reward_ckpts/UnifiedReward-2.0-qwen35-4b/."""
    _unifiedreward_qwen35("4b")


def unifiedreward_qwen35_9b():
    """UnifiedReward-2.0-qwen35-9b (pinned, ~18 GB) -> reward_ckpts/UnifiedReward-2.0-qwen35-9b/."""
    _unifiedreward_qwen35("9b")


def unifiedreward_qwen35_27b():
    """UnifiedReward-2.0-qwen35-27b (pinned, ~51 GB) -> reward_ckpts/UnifiedReward-2.0-qwen35-27b/."""
    _unifiedreward_qwen35("27b")





# --------------------------------------------------------------------------- #

def aesthetic():
    """LAION aesthetic MLP -> reward_ckpts/ (~4 MB)."""
    name = "sac+logos+ava1-l14-linearMSE.pth"
    url = ("https://github.com/christophschuhmann/improved-aesthetic-predictor"
           f"/raw/main/{name}")
    os.makedirs(REWARD_CKPTS, exist_ok=True)
    dest = os.path.join(REWARD_CKPTS, name)
    if not os.path.exists(dest):
        print(f"  ↓ {url}")
        urllib.request.urlretrieve(url, dest)
    print(f"    -> {dest}")


MODELS = {
    "stable_diffusion_3": stable_diffusion_3,
    "stable_diffusion_3_large": stable_diffusion_3_large,
    "flux_dev": flux_dev,
    "clip": clip,
    "dfn_clip": dfn_clip,
    "hpsv2": hpsv2,
    "aesthetic": aesthetic,
}
# Extension reward models: requested by name only, never by --all.
EXTENSION_MODELS = {
    "hpsv3": hpsv3,
    "unifiedreward_qwen35_4b": unifiedreward_qwen35_4b,
    "unifiedreward_qwen35_9b": unifiedreward_qwen35_9b,
    "unifiedreward_qwen35_27b": unifiedreward_qwen35_27b,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*",
                        help="which models to download (see --list)")
    parser.add_argument("--all", action="store_true", help="download everything")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    args = parser.parse_args()

    available = {**MODELS, **EXTENSION_MODELS}
    unknown = [m for m in args.models if m not in available]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)} — see --list")

    if args.list or (not args.models and not args.all):
        print(f"HF_HOME = {os.environ.get('HF_HOME', '~/.cache/huggingface (default!)')}")
        print(f"reward_ckpts -> {REWARD_CKPTS}\n\nAvailable models:")
        for name, fn in MODELS.items():
            print(f"  {name:<26} {fn.__doc__.splitlines()[0]}")
        print("\nExtension reward models (by name only; not included in --all):")
        for name, fn in EXTENSION_MODELS.items():
            print(f"  {name:<26} {fn.__doc__.splitlines()[0]}")
        print("\nUsage: python scripts/download_weights.py --all   (or a list of names)")
        return

    if not os.environ.get("HF_HOME"):
        print("WARNING: HF_HOME is not set — downloading into ~/.cache/huggingface.\n"
              "         Set HF_HOME to use a different cache directory.\n")

    names = list(MODELS) if args.all else args.models
    failed = []
    for name in names:
        print(f"\n=== {name} ===")
        try:
            available[name]()
        except Exception as e:  # keep going; report at the end
            msg = str(e)
            if "401" in msg or "403" in msg or "gated" in msg.lower():
                msg += "\n    -> GATED repo: accept the license on the Hub page " \
                       "and `hf auth login` first."
            print(f"  FAILED {name}: {type(e).__name__}: {msg}")
            failed.append(name)

    print("\n" + "=" * 60)
    done = [n for n in names if n not in failed]
    print(f"done: {', '.join(done) if done else '(none)'}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("\nFor offline use:\n"
          f"  export HF_HOME={os.environ.get('HF_HOME', '/path/to/hf_cache')}\n"
          "  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
