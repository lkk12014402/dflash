#!/usr/bin/env bash
# =============================================================================
# merge_data.sh — merge sharded generation outputs into one training file,
# dropping malformed/empty rows. Reports final count and token stats.
#
# Usage:
#   bash scripts/merge_data.sh
#   OUT_DIR=cache/full MERGED=cache/full/train_all.jsonl bash scripts/merge_data.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-cache/full}"
MERGED="${MERGED:-cache/full/train_all.jsonl}"

shopt -s nullglob
parts=("$OUT_DIR"/train_part*.jsonl)
if [ "${#parts[@]}" -eq 0 ]; then
  echo "[merge] no shard files in $OUT_DIR/train_part*.jsonl"; exit 1
fi
echo "[merge] merging ${#parts[@]} shard(s) -> $MERGED"

python - "$MERGED" "${parts[@]}" <<'PY'
import json, sys
merged, parts = sys.argv[1], sys.argv[2:]
kept = dropped = 0
lens = []
with open(merged, "w") as out:
    for p in parts:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                ids = r["input_ids"]; pl = r["prompt_len"]
            except Exception:
                dropped += 1; continue
            if not isinstance(ids, list) or len(ids) <= pl or pl <= 0:
                dropped += 1; continue
            out.write(json.dumps({"input_ids": ids, "prompt_len": pl}) + "\n")
            kept += 1
            lens.append((len(ids), pl))
print(f"[merge] kept={kept} dropped={dropped}")
if lens:
    import statistics as st
    tot = [a for a, _ in lens]; gen = [a - b for a, b in lens]
    print(f"[merge] total_len  mean={st.mean(tot):.1f} max={max(tot)}")
    print(f"[merge] gen_tokens mean={st.mean(gen):.1f} min={min(gen)} max={max(gen)}")
PY

echo "[merge] DONE -> $MERGED"
echo "[merge] Next: DATA=$MERGED bash scripts/train_full.sh"
