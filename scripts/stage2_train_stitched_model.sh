#!/usr/bin/env bash
# Stage 2 — Value-model distillation from locally fitted stitching adapters.
# See docs/STAGE2_VALUE_MODEL_DISTILLATION.md for setup, configuration, and output files.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/paths.sh
source scripts/stage2_config.sh

run_training() {
  local backbone=$1 reward=$2
  shift 2
  configure_training_recipe "$backbone" "$reward"
  local command=("${LAUNCH[@]}" --diffusion_model "$backbone"
                 --feedforward_model "$reward" "${TRAINING_ARGS[@]}" "$@")
  printf '%s: LR=%s; global batch=%s (%s processes × %s images/device × %s accumulation)\n' \
    "$recipe" "$TRAIN_LEARNING_RATE" "$TRAIN_GLOBAL_BATCH" "$NPROC" "$TRAIN_MICRO_BATCH" "$TRAIN_GA"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

stage2_hpsv2() {
  run_training sd3.5 hpsv2 \
    --hpsv2_checkpoint_dir "$STITCHVM_REWARD_CKPTS/hpsv2" \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_1_dit_3 \
    --initialization_weight_path "$INIT/hpsv2_sd3.5_dit_multires/state_dict_dit_blocks_3_enc_blocks_1_post.pt" \
    --save_path "$OUT/stitchvm_hpsv2_sd3"
}

stage2_clip() {
  run_training sd3.5 clip \
    --clip_model_id openai/clip-vit-large-patch14-336 \
    --feedforward_width 336 --feedforward_height 336 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_3 \
    --initialization_weight_path "$INIT/clip_sd3.5_dit_multires/state_dict_dit_blocks_3_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_clip_sd3"
}

stage2_aesthetic() {
  run_training sd3.5 aesthetic \
    --aesthetic_mlp_path "$STITCHVM_REWARD_CKPTS/sac+logos+ava1-l14-linearMSE.pth" \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_3 \
    --initialization_weight_path "$INIT/aesthetic_sd3.5_dit_multires/state_dict_dit_blocks_3_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_aesthetic_sd3"
}

stage2_dfn_clip() {
  run_training sd3.5 dfn_clip \
    --dfn_clip_model_id hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --feedforward_width 378 --feedforward_height 378 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_5 \
    --initialization_weight_path "$INIT/dfn_clip_sd3.5_dit_multires/state_dict_dit_blocks_5_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_dfn_clip_sd3"
}

stage2_large_hpsv2() {
  run_training sd3.5_large hpsv2 \
    --hpsv2_checkpoint_dir "$STITCHVM_REWARD_CKPTS/hpsv2" \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_1_dit_13 \
    --initialization_weight_path "$INIT/hpsv2_sd3.5_large_dit_multires/state_dict_dit_blocks_13_enc_blocks_1_post.pt" \
    --save_path "$OUT/stitchvm_hpsv2_sd3_large"
}

stage2_large_clip() {
  run_training sd3.5_large clip \
    --clip_model_id openai/clip-vit-large-patch14-336 \
    --feedforward_width 336 --feedforward_height 336 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_12 \
    --initialization_weight_path "$INIT/clip_sd3.5_large_dit_multires/state_dict_dit_blocks_12_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_clip_sd3_large"
}

stage2_large_aesthetic() {
  run_training sd3.5_large aesthetic \
    --aesthetic_mlp_path "$STITCHVM_REWARD_CKPTS/sac+logos+ava1-l14-linearMSE.pth" \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_12 \
    --initialization_weight_path "$INIT/aesthetic_sd3.5_large_dit_multires/state_dict_dit_blocks_12_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_aesthetic_sd3_large"
}

stage2_large_dfn_clip() {
  run_training sd3.5_large dfn_clip \
    --dfn_clip_model_id hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --feedforward_width 378 --feedforward_height 378 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_15 \
    --initialization_weight_path "$INIT/dfn_clip_sd3.5_large_dit_multires/state_dict_dit_blocks_15_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_dfn_clip_sd3_large"
}

stage2_flux_hpsv2() {
  run_training flux_dev hpsv2 \
    --hpsv2_checkpoint_dir "$STITCHVM_REWARD_CKPTS/hpsv2" \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_1_dit_9 \
    --initialization_weight_path "$INIT/hpsv2_flux_dev_dit_multires/state_dict_dit_blocks_9_enc_blocks_1_post.pt" \
    --save_path "$OUT/stitchvm_hpsv2_flux"
}

stage2_flux_clip() {
  run_training flux_dev clip \
    --clip_model_id openai/clip-vit-large-patch14-336 \
    --feedforward_width 336 --feedforward_height 336 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_9 \
    --initialization_weight_path "$INIT/clip_flux_dev_dit_multires/state_dict_dit_blocks_9_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_clip_flux"
}

stage2_flux_aesthetic() {
  run_training flux_dev aesthetic \
    --aesthetic_mlp_path "$STITCHVM_REWARD_CKPTS/sac+logos+ava1-l14-linearMSE.pth" \
    --feedforward_width 224 --feedforward_height 224 \
    --stitching_layer_config conv2d_k1x1_o1024_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_9 \
    --initialization_weight_path "$INIT/aesthetic_flux_dev_dit_multires/state_dict_dit_blocks_9_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_aesthetic_flux"
}

stage2_flux_dfn_clip() {
  run_training flux_dev dfn_clip \
    --dfn_clip_model_id hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --feedforward_width 378 --feedforward_height 378 \
    --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \
    --stitching_layer_location enc_blocks_2_dit_9 \
    --initialization_weight_path "$INIT/dfn_clip_flux_dev_dit_multires/state_dict_dit_blocks_9_enc_blocks_2_post.pt" \
    --save_path "$OUT/stitchvm_dfn_clip_flux"
}

RECIPES=(stage2_hpsv2 stage2_clip stage2_aesthetic stage2_dfn_clip stage2_large_hpsv2 stage2_large_clip stage2_large_aesthetic stage2_large_dfn_clip stage2_flux_hpsv2 stage2_flux_clip stage2_flux_aesthetic stage2_flux_dfn_clip)
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/stage2_train_stitched_model.sh <recipe> [recipe ...]"
  echo "Set DRY_RUN=1 to print resolved settings and commands without training."
  echo "Optional: NPROC, MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE (accumulation is derived)."
  printf "  %s\n" "${RECIPES[@]}"
  exit 0
fi

DRY_RUN=${DRY_RUN:-0}
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1." >&2; exit 2
fi
for recipe in "$@"; do
  case " ${RECIPES[*]} " in
    *" $recipe "*) ;;
    *) echo "Unknown Stage 2 recipe: $recipe" >&2; exit 2 ;;
  esac
done

configure_training_launcher
build_launcher train_stitched_model.py
for recipe in "$@"; do
  "$recipe"
done
