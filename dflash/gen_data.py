"""Generate target-greedy training data for DFlash.

The DFlash training objective is only inference-equivalent when the sequences it
trains on lie on the *target's own* decoding trajectory (see the docstring in
``train.py``). The cleanest way to guarantee that for a temperature-0 verifier
is to let the target greedily continue each prompt and train hard cross-entropy
on those tokens -- equivalent to distilling the greedy verifier.

This script takes prompts (from the built-in benchmark datasets or a custom
jsonl of ``{"prompt": ...}`` / ``{"messages": [...]}``) and writes a jsonl of
``{"input_ids": [...], "prompt_len": P}`` that ``train.py`` consumes directly.

Example
-------
    CUDA_VISIBLE_DEVICES=0 python -m dflash.gen_data \
        --model Qwen/Qwen3-4B --dataset gsm8k --max-samples 2000 \
        --max-new-tokens 1024 --out cache/train_gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .benchmark import DATASETS, load_and_process_dataset, _apply_chat_template


def _prompts_from_source(source: str, cap: int) -> list[str]:
    """Return a list of raw user-prompt strings from a named data source."""
    from datasets import load_dataset
    from huggingface_hub import get_token

    out: list[str] = []
    if source == "codealpaca":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        for r in ds:
            instr = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            if not instr:
                continue
            out.append(f"{instr}\n\n{inp}" if inp else instr)
            if len(out) >= cap:
                break
    elif source == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for r in ds:
            msgs = r.get("messages") or []
            first = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            if first:
                out.append(first.strip())
            if len(out) >= cap:
                break
    elif source == "nemotron-v2":
        # Gated dataset; no "train" split — it is partitioned by category.
        # Pull a balanced mix across the non-multilingual reasoning splits.
        cats = ["chat", "code", "math", "stem"]
        per_cat = max(1, cap // len(cats))
        for ci, cat in enumerate(cats):
            ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=cat,
                              streaming=True, token=get_token())
            taken = 0
            for r in ds:
                msgs = r.get("messages") or r.get("conversations") or []
                first = None
                for m in msgs:
                    role = m.get("role") or m.get("from")
                    if role in ("user", "human"):
                        first = m.get("content") or m.get("value")
                        break
                if first and first.strip():
                    out.append(first.strip())
                    taken += 1
                # last category absorbs the remainder to reach cap
                limit = cap if ci == len(cats) - 1 else per_cat
                if taken >= per_cat and len(out) >= limit:
                    break
                if len(out) >= cap:
                    break
            if len(out) >= cap:
                break
    else:
        raise ValueError(f"unknown source '{source}'")
    return out


def _collect_prompts(args, tokenizer):
    """Build the final list of chat-templated prompt strings (after sharding)."""
    if args.prompts_jsonl:
        with open(args.prompts_jsonl) as f:
            rows = [json.loads(line) for line in f]
        raw = []
        for r in rows:
            if "messages" in r:
                raw.append(r["messages"])
            else:
                raw.append([{"role": "user", "content": r["prompt"]}])
    elif args.source:
        import random as _r
        sources = [s.strip() for s in args.source.split(",") if s.strip()]
        pooled: list[str] = []
        for s in sources:
            got = _prompts_from_source(s, args.per_source)
            print(f"[gen_data] source {s}: {len(got)} prompts")
            pooled.extend(got)
        _r.Random(args.seed).shuffle(pooled)
        raw = [[{"role": "user", "content": p}] for p in pooled]
    else:
        data = load_and_process_dataset(args.dataset)
        raw = [[{"role": "user", "content": item["turns"][0]}] for item in data]

    if args.max_samples:
        raw = raw[: args.max_samples]
    # shard across parallel generation processes (one per GPU)
    raw = raw[args.shard_id :: args.num_shards]
    return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=args.enable_thinking) for m in raw]


@torch.inference_mode()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description="Generate target-greedy DFlash training data")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="gsm8k", choices=list(DATASETS.keys()))
    p.add_argument("--source", default=None,
                   help="comma list of {codealpaca,ultrachat,nemotron-v2} (overrides --dataset)")
    p.add_argument("--per-source", type=int, default=10000, help="max prompts per source")
    p.add_argument("--prompts-jsonl", default=None, help="custom prompts instead of --dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    prompts = _collect_prompts(args, tokenizer)
    print(f"[gen_data] shard {args.shard_id}/{args.num_shards}: {len(prompts)} prompts to generate")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            chunk = prompts[i:i + args.batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            gen = target.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = enc["input_ids"].shape[1]      # left-padded -> shared length
            for b in range(gen.shape[0]):
                full = gen[b].tolist()
                # strip left padding of this row
                row_attn = enc["attention_mask"][b]
                pad = int((row_attn == 0).sum().item())
                full = full[pad:]
                p_len = prompt_len - pad
                # drop trailing pad/eos padding from generation
                while len(full) > p_len and full[-1] == tokenizer.pad_token_id:
                    full.pop()
                if len(full) - p_len < 2:
                    continue
                fout.write(json.dumps({"input_ids": full, "prompt_len": p_len}) + "\n")
                n += 1
    print(f"[gen_data] wrote {n} sequences -> {out_path}")


if __name__ == "__main__":
    main()
