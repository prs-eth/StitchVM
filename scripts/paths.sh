# Shared paths and data-loading settings. Override through the environment.
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export STITCHVM_REWARD_CKPTS=${STITCHVM_REWARD_CKPTS:-reward_ckpts}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DATA=${DATA:-data}                        # datasets (see README)
INIT=${INIT:-init}                        # Stage 1 initializations
OUT=${OUT:-outputs}                      # Stage 2 checkpoints
HPD="HPDv2:$DATA/hpdv2"
AVA="ava:$DATA/aesthetic_benchmark/ava"
NUM_WORKERS=${NUM_WORKERS:-3}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}

# NPROC counts processes across all machines. Configure only for Stage 2.
configure_training_launcher() {
  NPROC=${NPROC:-1}
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  MAIN_IP=${MAIN_IP:-127.0.0.1}
  MAIN_PORT=${MAIN_PORT:-29500}
  if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ || ! "$NNODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC and NNODES must be positive integers." >&2; return 2
  fi
  if (( NPROC % NNODES != 0 )); then
    echo "NPROC must be divisible by NNODES." >&2; return 2
  fi
  if [[ ! "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); then
    echo "NODE_RANK must be in [0, NNODES)." >&2; return 2
  fi
  if (( NNODES > 1 )) && [[ "$MAIN_IP" == 127.0.0.1 ]]; then
    echo "Set MAIN_IP to the coordinator's reachable address for multiple machines." >&2; return 2
  fi
}

build_launcher() {
  local script=$1
  if [[ "$NNODES" -gt 1 ]]; then
    LAUNCH=(accelerate launch --multi_gpu
            --num_processes "$NPROC" --num_machines "$NNODES"
            --machine_rank "$NODE_RANK"
            --main_process_ip "$MAIN_IP" --main_process_port "$MAIN_PORT"
            --mixed_precision bf16 "$script")
  elif [[ "$NPROC" -gt 1 ]]; then
    LAUNCH=(accelerate launch --multi_gpu --num_processes "$NPROC"
            --mixed_precision bf16 "$script")
  else
    LAUNCH=(python "$script")
  fi
}

# Optional memory saving: encode one image at a time in the VAE.
export STITCHVM_VAE_SLICING=${STITCHVM_VAE_SLICING:-0}

# Avoid inherited UCX error handlers when DataLoader workers are forked.
export UCX_HANDLE_ERRORS=${UCX_HANDLE_ERRORS:-none}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-error}
