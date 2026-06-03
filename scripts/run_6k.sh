#!/usr/bin/env bash
# =============================================================================
# run_6k.sh — reproduce the 6000-sample scaled run we did (gen -> merge -> train).
#
# This is the small "proof-of-correctness" run, NOT the full 800K recipe.
# It generates 6000 target-greedy sequences from codealpaca+ultrachat across
# GPUs 4-7, merges them, then trains the draft FRESH from config for 4 epochs.
# Expect loss 8.2 -> ~5.6 and acc@1 to climb — quality is far below the released
# model by design (2 orders of magnitude less data). Use run scripts gen_full.sh
# / train_full.sh for the real full-scale run.
#
# Usage:
#   bash scripts/run_6k.sh                 # gen + merge + train, GPUs 4-7
#   SKIP_GEN=1 bash scripts/run_6k.sh      # reuse existing cache, just train
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
GPUS="${GPUS:-4,5,6,7}"
OUT_DIR="${OUT_DIR:-cache/run6k}"
MERGED="${MERGED:-cache/run6k/train_all.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-dflash_ckpt_scratch}"
SKIP_GEN="${SKIP_GEN:-0}"

# ---- 1) generate 6000 sequences (1500 per GPU) -----------------------------
if [ "$SKIP_GEN" != "1" ]; then
  GPUS="$GPUS" MODEL="$MODEL" \
    SOURCE="codealpaca,ultrachat" \
    PER_SOURCE=3000 MAX_SAMPLES=6000 \
    MAX_NEW_TOKENS=384 BATCH_SIZE=24 SEED=0 \
    OUT_DIR="$OUT_DIR" LOG_DIR="logs/run6k" \
    bash scripts/gen_full.sh
fi

# ---- 2) merge --------------------------------------------------------------
OUT_DIR="$OUT_DIR" MERGED="$MERGED" bash scripts/merge_data.sh

# ---- 3) train fresh-from-config, 4 epochs, wandb ---------------------------
DATA="$MERGED" MODEL="$MODEL" GPUS="$GPUS" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  MAX_LEN=2048 BATCH_SIZE=4 GRAD_ACCUM=4 EPOCHS=4 \
  LR=1e-4 LOSS_DECAY=16 LOG_EVERY=10 SAVE_EVERY=0 \
  WANDB=1 WANDB_PROJECT=dflash-draft WANDB_RUN_NAME=scratch-b16-6k \
  MASTER_PORT=29541 \
  bash scripts/train_full.sh

echo "[run_6k] launched. Final ckpt -> $OUTPUT_DIR/final  (tail -f logs/scratch-b16-6k.log)"
