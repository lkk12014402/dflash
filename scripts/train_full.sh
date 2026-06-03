#!/usr/bin/env bash
# =============================================================================
# train_full.sh — full DDP training of the DFlash draft model on GPUs 4-7.
#
# Trains FRESH from the published draft config (z-lab/Qwen3-4B-DFlash-b16) by
# default. Set INIT_FROM=<path> to continue from an existing draft instead.
# Logs to Weights & Biases (set WANDB=0 to disable).
#
# Usage:
#   DATA=cache/full/train_all.jsonl bash scripts/train_full.sh
#   GPUS="0,1,2,3" EPOCHS=2 BATCH_SIZE=8 bash scripts/train_full.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- config (override via env) ---------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATA="${DATA:-cache/full/train_all.jsonl}"
DRAFT_CONFIG="${DRAFT_CONFIG:-z-lab/Qwen3-4B-DFlash-b16}"
INIT_FROM="${INIT_FROM:-}"             # set to continue from a draft ckpt
OUTPUT_DIR="${OUTPUT_DIR:-dflash_ckpt_full}"
GPUS="${GPUS:-4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29540}"

MAX_LEN="${MAX_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
LOSS_DECAY="${LOSS_DECAY:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-2000}"       # periodic ckpt for long runs (0=off)

# optional logit-distillation (KL) — default off (pure CE on target tokens)
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.0}"
DISTILL_TEMP="${DISTILL_TEMP:-1.0}"

# wandb
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-dflash-draft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-full-b16}"

if [ ! -f "$DATA" ]; then
  echo "[train_full] data not found: $DATA (run gen_full.sh + merge_data.sh first)"; exit 1
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC="${#GPU_ARR[@]}"
mkdir -p logs

# init: fresh-from-config OR continue-from-checkpoint
if [ -n "$INIT_FROM" ]; then
  INIT_ARGS=(--init-from "$INIT_FROM")
  echo "[train_full] continuing from $INIT_FROM"
else
  INIT_ARGS=(--draft-config "$DRAFT_CONFIG")
  echo "[train_full] fresh from config $DRAFT_CONFIG"
fi

WANDB_ARGS=()
if [ "$WANDB" = "1" ]; then
  WANDB_ARGS=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
fi

LOG="logs/${WANDB_RUN_NAME}.log"
echo "[train_full] gpus=$GPUS nproc=$NPROC data=$DATA out=$OUTPUT_DIR"
echo "[train_full] bs=$BATCH_SIZE ga=$GRAD_ACCUM epochs=$EPOCHS lr=$LR decay=$LOSS_DECAY"
echo "[train_full] logging to $LOG  (tail -f $LOG)"

CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="$PYTHONPATH" \
  nohup torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m dflash.train train \
      --model "$MODEL" \
      --data "$DATA" \
      "${INIT_ARGS[@]}" \
      --output-dir "$OUTPUT_DIR" \
      --max-len "$MAX_LEN" \
      --batch-size "$BATCH_SIZE" \
      --grad-accum "$GRAD_ACCUM" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --loss-decay "$LOSS_DECAY" \
      --warmup-ratio "$WARMUP_RATIO" \
      --num-workers "$NUM_WORKERS" \
      --log-every "$LOG_EVERY" \
      --save-every "$SAVE_EVERY" \
      --distill-weight "$DISTILL_WEIGHT" \
      --distill-temp "$DISTILL_TEMP" \
      "${WANDB_ARGS[@]}" \
      > "$LOG" 2>&1 &

echo "[train_full] launched pid $! (detached). Final ckpt -> $OUTPUT_DIR/final"
