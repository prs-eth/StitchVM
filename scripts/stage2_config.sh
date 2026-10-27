# Shared Stage 2 training settings. Sourced after scripts/paths.sh.
# Recipe functions keep model-specific layers, checkpoints, and output paths.
configure_training_recipe() {
  local backbone=$1 reward=$2
  local default_global_batch=256 default_micro_batch=16
  local buckets=(512 768 1024 768x1024 1024x768)
  local loss_type=mse

  TRAIN_LEARNING_RATE=5e-5
  TRAIN_ENV=()
  case "$backbone" in
    sd3.5) ;;
    sd3.5_large) TRAIN_LEARNING_RATE=1e-5 ;;
    flux_dev) default_global_batch=128; default_micro_batch=8 ;;
    *) echo "Unknown training backbone: $backbone" >&2; return 2 ;;
  esac
  case "$reward" in
    hpsv2|clip) ;;
    aesthetic)
      if [[ "$backbone" != flux_dev ]]; then loss_type=l1; fi
      ;;
    dfn_clip)
      default_global_batch=128
      default_micro_batch=8
      buckets=(512 768 1024)
      ;;
    hpsv3|qwen35)
      # Vision-language reward heads (SD3.5 Medium only): post-merger token MSE,
      # global batch 256 at micro-batch 8. expandable_segments avoids the
      # allocator fragmentation seen at the 1024² buckets.
      if [[ "$backbone" != sd3.5 ]]; then
        echo "$reward recipes are defined for SD3.5 Medium only" >&2; return 2
      fi
      default_micro_batch=8
      TRAIN_ENV=(env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}")
      ;;
    *) echo "Unknown training reward encoder: $reward" >&2; return 2 ;;
  esac

  # Keep input overrides separate from resolved values so sequential recipes
  # each start from their own defaults.
  TRAIN_GLOBAL_BATCH=${GLOBAL_BATCH_SIZE-$default_global_batch}
  TRAIN_MICRO_BATCH=${MICRO_BATCH_SIZE-$default_micro_batch}
  if [[ ! "$TRAIN_GLOBAL_BATCH" =~ ^[1-9][0-9]*$ || ! "$TRAIN_MICRO_BATCH" =~ ^[1-9][0-9]*$ ]]; then
    echo "GLOBAL_BATCH_SIZE and MICRO_BATCH_SIZE must be positive integers." >&2; return 2
  fi
  local synchronized_batch=$((NPROC * TRAIN_MICRO_BATCH))
  if (( TRAIN_GLOBAL_BATCH % synchronized_batch != 0 )); then
    echo "Global batch $TRAIN_GLOBAL_BATCH must be divisible by NPROC ($NPROC) × micro-batch ($TRAIN_MICRO_BATCH). Lower MICRO_BATCH_SIZE to preserve the target." >&2
    return 2
  fi
  TRAIN_GA=$((TRAIN_GLOBAL_BATCH / synchronized_batch))
  if [[ ${GA+x} ]]; then
    if [[ ! "$GA" =~ ^[1-9][0-9]*$ ]] || (( GA != TRAIN_GA )); then
      echo "GA must equal the derived accumulation ($TRAIN_GA). Set MICRO_BATCH_SIZE or GLOBAL_BATCH_SIZE to change the batch configuration." >&2
      return 2
    fi
  fi

  TRAINING_ARGS=(
    --dataset "$HPD" --dataset "$AVA"
    --num_epochs 5 --width 512 --height 512
    --resolution_buckets "${buckets[@]}"
    --batch_size "$TRAIN_MICRO_BATCH" --gradient_accumulation_steps "$TRAIN_GA"
    --learning_rate "$TRAIN_LEARNING_RATE" --stitching_lr_multiplier 1.0
    --weight_decay 0.0 --warmup_steps 100
    --timestep_range 0.0,1.0 --timestep_distribution center
    --ema_decay 0.999 --optimize_speed --loss_type "$loss_type"
    --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR"
  )
  if [[ "$reward" == aesthetic ]]; then
    TRAINING_ARGS+=(--score_weight 0.1)
  fi
}
