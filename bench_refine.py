"""Compare reference one-shot DFlash draft vs iterative self-refinement (Innovation 2).

Runs the reference ``dflash_generate`` (num_refine=0 equivalent) against
``dflash_generate_refine`` for one or more ``--num-refine`` settings on the same
greedy prompts. Reports throughput, mean acceptance length, target forwards
(NFE), draft forwards, tokens / target-forward, and losslessness vs reference.

Example:
  python bench_refine.py --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 \
      --dataset gsm8k --max-samples 16 --max-new-tokens 256 \
      --num-refine 1 2 --refine-threshold 0.6
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.model import DFlashDraftModel, dflash_generate
from dflash.refine import dflash_generate_refine
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
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--num-refine", type=int, nargs="+", default=[1])
    p.add_argument("--refine-threshold", type=float, default=0.6)
    args = p.parse_args()

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
    print(f"[cfg] base_block={base_block} num_refine={args.num_refine} "
          f"refine_threshold={args.refine_threshold}")

    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples)

    def build_ids(user_content):
        text = _apply_chat_template(tokenizer, [{"role": "user", "content": user_content}], args.enable_thinking)
        return tokenizer.encode(text, return_tensors="pt").to(device)

    eos = [tokenizer.eos_token_id]

    def new_acc():
        return {"tpot": [], "accept": [], "nfe": 0, "draft_fwd": 0, "out_tok": 0}

    # warmup
    warm = build_ids("Hi")
    dflash_generate(draft, target=target, input_ids=warm, max_new_tokens=8,
                    stop_token_ids=eos, temperature=0.0, block_size=base_block, return_stats=True)
    for nr in args.num_refine:
        dflash_generate_refine(draft, target=target, input_ids=warm, max_new_tokens=8,
                               stop_token_ids=eos, temperature=0.0, block_size=base_block,
                               num_refine=nr, refine_threshold=args.refine_threshold)

    ref = new_acc()
    variants = {nr: new_acc() for nr in args.num_refine}
    mismatch = {nr: 0 for nr in args.num_refine}
    compared = 0

    for i, inst in enumerate(dataset):
        ids = build_ids(inst["turns"][0])

        rf = dflash_generate(
            draft, target=target, input_ids=ids, max_new_tokens=args.max_new_tokens,
            stop_token_ids=eos, temperature=args.temperature,
            block_size=base_block, return_stats=True,
        )
        ref["tpot"].append(rf.time_per_output_token)
        ref["accept"].append(np.mean(rf.acceptance_lengths))
        ref["nfe"] += len(rf.acceptance_lengths)
        ref["draft_fwd"] += len(rf.acceptance_lengths)
        ref["out_tok"] += rf.num_output_tokens
        a = rf.output_ids[0, rf.num_input_tokens:].tolist()
        compared += 1

        for nr in args.num_refine:
            rr = dflash_generate_refine(
                draft, target=target, input_ids=ids, max_new_tokens=args.max_new_tokens,
                stop_token_ids=eos, temperature=args.temperature,
                block_size=base_block, num_refine=nr, refine_threshold=args.refine_threshold,
            )
            v = variants[nr]
            v["tpot"].append(rr.time_per_output_token)
            v["accept"].append(np.mean(rr.acceptance_lengths))
            v["nfe"] += rr.num_target_forwards
            v["draft_fwd"] += rr.num_draft_forwards
            v["out_tok"] += rr.num_output_tokens
            if args.temperature == 0.0:
                b = rr.output_ids[0, rr.num_input_tokens:].tolist()
                n = min(len(a), len(b))
                if a[:n] != b[:n]:
                    mismatch[nr] += 1

        if (i + 1) % 4 == 0:
            msg = f"  [{i+1}/{len(dataset)}] ref {1/np.mean(ref['tpot']):.1f} tok/s acc={np.mean(ref['accept']):.2f}"
            for nr in args.num_refine:
                v = variants[nr]
                msg += f" | r{nr} {1/np.mean(v['tpot']):.1f} tok/s acc={np.mean(v['accept']):.2f}"
            print(msg)

    def tps(acc):
        return 1.0 / float(np.mean(acc["tpot"]))

    cols = ["reference"] + [f"refine={nr}" for nr in args.num_refine]
    accs = [ref] + [variants[nr] for nr in args.num_refine]
    w = 14
    print("\n" + "=" * (30 + w * len(cols)))
    print(f"{'metric':<30}" + "".join(f"{c:>{w}}" for c in cols))
    print("-" * (30 + w * len(cols)))

    def row(name, fn):
        print(f"{name:<30}" + "".join(f"{fn(a):>{w}}" for a in accs))

    row("throughput (tok/s)", lambda a: f"{tps(a):.2f}")
    row("mean acceptance length", lambda a: f"{np.mean(a['accept']):.2f}")
    row("target forwards (NFE)", lambda a: f"{a['nfe']:d}")
    row("draft forwards", lambda a: f"{a['draft_fwd']:d}")
    row("output tokens (total)", lambda a: f"{a['out_tok']:d}")
    row("tokens / target-forward", lambda a: f"{a['out_tok']/max(a['nfe'],1):.2f}")
    print("-" * (30 + w * len(cols)))
    base = tps(ref)
    for nr in args.num_refine:
        print(f"decode speedup refine={nr} / reference: {tps(variants[nr]) / base:.3f}x")
    if args.temperature == 0.0:
        for nr in args.num_refine:
            print(f"losslessness refine={nr}: {compared - mismatch[nr]}/{compared} token-identical to reference")
    print("=" * (30 + w * len(cols)))


if __name__ == "__main__":
    main()
