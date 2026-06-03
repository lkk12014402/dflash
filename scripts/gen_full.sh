#!/usr/bin/env bash
# =============================================================================
# gen_full.sh — multi-GPU sharded target-greedy data generation for DFlash.
#
# Launches one generation process per GPU. Each process loads the SAME global
# prompt pool (codealpaca + ultrachat + nemotron-v2), caps it to --max-samples,
# then takes its 1/N shard (shard_id :: num_shards). Identical SEED across
# shards is REQUIRED so the global pool is shuffled the same way everywhere.
#
# Output: cache/full/train_part{0..N-1}.jsonl  (+ logs/gen_full/gen{i}.log)
# Next:   scripts/merge_data.sh   then   scripts/train_full.sh
#
# Usage:
#   bash scripts/gen_full.sh                  # full run, GPUs 4-7, 800K target
#   GPUS="0,1,2,3" MAX_SAMPLES=100000 bash scripts/gen_full.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- config (override via env) ---------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-4B}"
GPUS="${GPUS:-4,5,6,7}"                 # comma list; one shard per GPU
SOURCE="${SOURCE:-codealpaca,ultrachat,nemotron-v2}"
MAX_SAMPLES="${MAX_SAMPLES:-800000}"   # GLOBAL cap across all shards
PER_SOURCE="${PER_SOURCE:-400000}"     # max prompts pulled per source
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SEED="${SEED:-0}"                      # MUST match across shards
OUT_DIR="${OUT_DIR:-cache/full}"
LOG_DIR="${LOG_DIR:-logs/gen_full}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_SHARDS="${#GPU_ARR[@]}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[gen_full] model=$MODEL gpus=$GPUS shards=$NUM_SHARDS"
echo "[gen_full] source=$SOURCE max_samples=$MAX_SAMPLES per_source=$PER_SOURCE"
echo "[gen_full] max_new_tokens=$MAX_NEW_TOKENS batch_size=$BATCH_SIZE seed=$SEED"
echo "[gen_full] out=$OUT_DIR logs=$LOG_DIR"

pids=()
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  out="$OUT_DIR/train_part${i}.jsonl"
  log="$LOG_DIR/gen${i}.log"
  echo "[gen_full] launch shard $i on GPU $gpu -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" \
    nohup python -m dflash.gen_data \
      --model "$MODEL" \
      --source "$SOURCE" \
      --per-source "$PER_SOURCE" \
      --max-samples "$MAX_SAMPLES" \
      --num-shards "$NUM_SHARDS" \
      --shard-id "$i" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --out "$out" \
      > "$log" 2>&1 &
  pids+=($!)
done

echo "[gen_full] pids: ${pids[*]}"
echo "[gen_full] waiting for all shards (tail -f $LOG_DIR/gen0.log to watch)..."
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; echo "[gen_full] PID $p FAILED"; fi
done

echo "[gen_full] ---- shard line counts ----"
for i in "${!GPU_ARR[@]}"; do
  f="$OUT_DIR/train_part${i}.jsonl"
  printf "  shard %d: %s rows  (%s)\n" "$i" "$(wc -l < "$f" 2>/dev/null || echo 0)" "$f"
done

if [ "$fail" -ne 0 ]; then
  echo "[gen_full] DONE WITH ERRORS — check $LOG_DIR/*.log"; exit 1
fi
echo "[gen_full] ALL DONE. Next: bash scripts/merge_data.sh"
