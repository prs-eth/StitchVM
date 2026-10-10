#!/usr/bin/env bash
# =============================================================================
# StitchVM Stage 1 — Stitching-layer search at multiple input resolutions.
#
# One function per (reward x backbone) search; run selectively:
#
#     bash scripts/stage1_find_stitching_layer.sh                # list steps
#     bash scripts/stage1_find_stitching_layer.sh stage1_clip    # run one
#     bash scripts/stage1_find_stitching_layer.sh stage1_clip stage1_flux_clip
#
# Override HF_HOME, DATA and INIT through the environment; defaults are in
# scripts/paths.sh. See docs/STAGE1_STITCHING_LAYER_SEARCH.md for setup and output filenames.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
source scripts/paths.sh   # configurable search paths

# =============================================================================
# 1. Stage 1 — Stitching-layer search (closed-form least-squares fits)
#    Output: $INIT/<reward>_<backbone>_dit_multires/state_dict_*_post.pt
#            + per-(dit,enc) MSE metrics. Output guide: docs/STAGE1_STITCHING_LAYER_SEARCH.md.
#            The script reports the grid; it does not automatically select a pair.
# =============================================================================

stage1_hpsv2() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/hpsv2_sd3.5_dit_multires" \
    --feedforward_model hpsv2 --diffusion_model sd3.5 \
    --hpsv2_checkpoint_dir "$STITCHVM_REWARD_CKPTS/hpsv2" \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_clip() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/clip_sd3.5_dit_multires" \
    --feedforward_model clip --diffusion_model sd3.5 \
    --clip_model_id openai/clip-vit-large-patch14-336 \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 336 --feedforward_height 336 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_aesthetic() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/aesthetic_sd3.5_dit_multires" \
    --feedforward_model aesthetic --diffusion_model sd3.5 \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_dfn_clip() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/dfn_clip_sd3.5_dit_multires" \
    --feedforward_model dfn_clip --diffusion_model sd3.5 \
    --dfn_clip_model_id hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --resolution_buckets 512 768 1024 \
    --feedforward_width 378 --feedforward_height 378 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

# --- SD 3.5-Large (distinct init/output paths) ---

stage1_large_hpsv2() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/hpsv2_sd3.5_large_dit_multires" \
    --feedforward_model hpsv2 --diffusion_model sd3.5_large \
    --hpsv2_checkpoint_dir "$STITCHVM_REWARD_CKPTS/hpsv2" \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_large_clip() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/clip_sd3.5_large_dit_multires" \
    --feedforward_model clip --diffusion_model sd3.5_large \
    --clip_model_id openai/clip-vit-large-patch14-336 \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 336 --feedforward_height 336 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_large_aesthetic() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/aesthetic_sd3.5_large_dit_multires" \
    --feedforward_model aesthetic --diffusion_model sd3.5_large \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_large_dfn_clip() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/dfn_clip_sd3.5_large_dit_multires" \
    --feedforward_model dfn_clip --diffusion_model sd3.5_large \
    --dfn_clip_model_id hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --resolution_buckets 512 768 1024 \
    --feedforward_width 378 --feedforward_height 378 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

# --- FLUX.1-dev search (HPDv2 + AVA, matching the SD3.5 sample budget) ---

stage1_flux_hpsv2() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/hpsv2_flux_dev_dit_multires" \
    --feedforward_model hpsv2 --diffusion_model flux_dev \
    --hpsv2_checkpoint_dir "$STITCHVM_REWARD_CKPTS/hpsv2" \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_flux_clip() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/clip_flux_dev_dit_multires" \
    --feedforward_model clip --diffusion_model flux_dev \
    --clip_model_id openai/clip-vit-large-patch14-336 \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 336 --feedforward_height 336 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_flux_aesthetic() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/aesthetic_flux_dev_dit_multires" \
    --feedforward_model aesthetic --diffusion_model flux_dev \
    --resolution_buckets 512 768 1024 768x1024 1024x768 \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

stage1_flux_dfn_clip() {
  python find_stitching_layer.py \
    --feature_save_path "$INIT/dfn_clip_flux_dev_dit_multires" \
    --feedforward_model dfn_clip --diffusion_model flux_dev \
    --dfn_clip_model_id hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --resolution_buckets 512 768 1024 \
    --feedforward_width 378 --feedforward_height 378 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --iterations_for_feature_extraction 300 --batch_size 1 --val_ratio 0.3334 \
    --timestep_range 0.0,1.0 \
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
    --dataset "$HPD" --dataset "$AVA"
}

# ----------------------------- dispatcher ------------------------------------
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/stage1_find_stitching_layer.sh <step> [step ...]"
  echo
  echo "Stage-1 searches: first 4 encoder blocks, first 50% of DiT blocks"
  echo "  SD 3.5-M, multires:  stage1_hpsv2 stage1_clip stage1_aesthetic stage1_dfn_clip"
  echo "  SD 3.5-L, multires:  stage1_large_hpsv2 stage1_large_clip stage1_large_aesthetic stage1_large_dfn_clip"
  echo "  FLUX, multires:  stage1_flux_hpsv2 stage1_flux_clip stage1_flux_aesthetic stage1_flux_dfn_clip"
  exit 0
fi
for step in "$@"; do
  echo "=== $step ==="
  "$step"
done
