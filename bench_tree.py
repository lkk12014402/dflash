"""Compare reference linear DFlash vs diffusion tree-drafting (Innovation 1).

Reports throughput, accepted-tokens-per-target-forward, target/draft forwards,
and losslessness vs the reference linear decoder, for one or more tree configs.

Example:
  python bench_tree.py --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 \
      --dataset gsm8k --max-samples 16 --max-new-tokens 256 \
      --config 32:3 48:3 64:4
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.model import DFlashDraftModel, dflash_generate
from dflash.tree import dflash_generate_tree
from dflash.benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template


def _attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--max-samples", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--config", nargs="+", default=["48:3"],
                   help="tree configs as max_nodes:top_k (e.g. 32:3 48:3 64:4)")
    args = p.parse_args()

    configs = []
    for c in args.config:
        n, k = c.split(":")
        configs.append((int(n), int(k)))

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    ai = _attn_impl()
    print(f"[load] target={args.model} draft={args.draft_model} attn={ai}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=ai, dtype=torch.bfloat16,
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model, attn_implementation=ai, dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    base_block = args.block_size if args.block_size is not None else draft.block_size
    print(f"[cfg] base_block={base_block} configs={configs}")

    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples)

    def build_ids(user_content):
        text = _apply_chat_template(tokenizer, [{"role": "user", "content": user_content}], args.enable_thinking)
        return tokenizer.encode(text, return_tensors="pt").to(device)

    eos = [tokenizer.eos_token_id]

    def new_acc():
        return {"tpot": [], "accept": [], "nfe": 0, "out_tok": 0, "nodes": []}

    warm = build_ids("Hi")
    dflash_generate(draft, target=target, input_ids=warm, max_new_tokens=8,
                    stop_token_ids=eos, temperature=0.0, block_size=base_block, return_stats=True)
    for (n, k) in configs:
        dflash_generate_tree(draft, target=target, input_ids=warm, max_new_tokens=8,
                             stop_token_ids=eos, temperature=0.0, block_size=base_block,
                             tree_max_nodes=n, top_k=k)

    ref = new_acc()
    variants = {c: new_acc() for c in configs}
    mismatch = {c: 0 for c in configs}
    compared = 0

    for i, inst in enumerate(dataset):
        ids = build_ids(inst["turns"][0])
        rf = dflash_generate(
            draft, target=target, input_ids=ids, max_new_tokens=args.max_new_tokens,
            stop_token_ids=eos, temperature=0.0, block_size=base_block, return_stats=True,
        )
        ref["tpot"].append(rf.time_per_output_token)
        ref["accept"].append(np.mean(rf.acceptance_lengths))
        ref["nfe"] += len(rf.acceptance_lengths)
        ref["out_tok"] += rf.num_output_tokens
        ref["nodes"].append(base_block)
        a = rf.output_ids[0, rf.num_input_tokens:].tolist()
        compared += 1

        for c in configs:
            n, k = c
            rt = dflash_generate_tree(
                draft, target=target, input_ids=ids, max_new_tokens=args.max_new_tokens,
                stop_token_ids=eos, temperature=0.0, block_size=base_block,
                tree_max_nodes=n, top_k=k,
            )
            v = variants[c]
            v["tpot"].append(rt.time_per_output_token)
            v["accept"].append(np.mean(rt.acceptance_lengths))
            v["nfe"] += rt.num_target_forwards
            v["out_tok"] += rt.num_output_tokens
            v["nodes"].extend(rt.tree_sizes)
            b = rt.output_ids[0, rt.num_input_tokens:].tolist()
            nn = min(len(a), len(b))
            if a[:nn] != b[:nn]:
                mismatch[c] += 1

        if (i + 1) % 4 == 0:
            msg = f"  [{i+1}/{len(dataset)}] ref {1/np.mean(ref['tpot']):.1f} tok/s a/f={np.mean(ref['accept']):.2f}"
            for c in configs:
                v = variants[c]
                msg += f" | {c[0]}:{c[1]} {1/np.mean(v['tpot']):.1f} tok/s a/f={np.mean(v['accept']):.2f}"
            print(msg)

    def tps(acc):
        return 1.0 / float(np.mean(acc["tpot"]))

    cols = ["reference"] + [f"tree {n}:{k}" for (n, k) in configs]
    accs = [ref] + [variants[c] for c in configs]
    w = 14
    width = 30 + w * len(cols)
    print("\n" + "=" * width)
    print(f"{'metric':<30}" + "".join(f"{c:>{w}}" for c in cols))
    print("-" * width)

    def row(name, fn):
        print(f"{name:<30}" + "".join(f"{fn(a):>{w}}" for a in accs))

    row("throughput (tok/s)", lambda a: f"{tps(a):.2f}")
    row("accepted tok / target-fwd", lambda a: f"{a['out_tok']/max(a['nfe'],1):.2f}")
    row("mean tree size (nodes)", lambda a: f"{np.mean(a['nodes']):.1f}")
    row("target forwards (NFE)", lambda a: f"{a['nfe']:d}")
    row("output tokens (total)", lambda a: f"{a['out_tok']:d}")
    print("-" * width)
    base = tps(ref)
    for c in configs:
        print(f"speedup tree {c[0]}:{c[1]} / reference: {tps(variants[c]) / base:.3f}x   "
              f"lossless: {compared - mismatch[c]}/{compared}")
    print("=" * width)


if __name__ == "__main__":
    main()
