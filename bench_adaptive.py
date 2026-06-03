"""Compare fixed-block DFlash vs Adaptive-Block Confidence-Guided DFlash.

Runs both decoders on the same prompts with greedy decoding, verifies
losslessness (token-id equality between the two decoders), and reports
throughput / mean acceptance length / number of target forwards (NFE) /
average drafted block size.

Example:
  python bench_adaptive.py --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 \
      --dataset gsm8k --max-samples 20 --max-new-tokens 512
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.model import DFlashDraftModel, dflash_generate
from dflash.adaptive import dflash_generate_adaptive
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
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--block-size", type=int, default=None, help="fixed/base block (default: draft.block_size)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true")
    # adaptive knobs
    p.add_argument("--min-block", type=int, default=4)
    p.add_argument("--max-block", type=int, default=None, help="default: 2x base block")
    p.add_argument("--conf-threshold", type=float, default=0.6)
    p.add_argument("--ema-beta", type=float, default=0.7)
    p.add_argument("--block-factor", type=float, default=1.5)
    p.add_argument("--no-adaptive-block", action="store_true")
    p.add_argument("--no-early-stop", action="store_true")
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
    max_block = args.max_block if args.max_block is not None else base_block * 2
    print(f"[cfg] base_block={base_block} max_block={max_block} "
          f"adaptive_block={not args.no_adaptive_block} early_stop={not args.no_early_stop} "
          f"conf_threshold={args.conf_threshold} block_factor={args.block_factor}")

    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples)

    def build_ids(user_content):
        text = _apply_chat_template(tokenizer, [{"role": "user", "content": user_content}], args.enable_thinking)
        return tokenizer.encode(text, return_tensors="pt").to(device)

    # warmup
    warm = build_ids("Hi")
    dflash_generate(draft, target=target, input_ids=warm, max_new_tokens=8,
                    stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                    block_size=base_block, return_stats=True)
    dflash_generate_adaptive(draft, target=target, input_ids=warm, max_new_tokens=8,
                             stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                             block_size=base_block, max_block=max_block)

    fixed = {"tpot": [], "accept": [], "nfe": 0, "out_tok": 0, "blocks": []}
    adapt = {"tpot": [], "accept": [], "nfe": 0, "out_tok": 0, "blocks": []}
    mismatches = 0
    compared = 0

    for i, inst in enumerate(dataset):
        ids = build_ids(inst["turns"][0])

        rf = dflash_generate(
            draft, target=target, input_ids=ids, max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id], temperature=args.temperature,
            block_size=base_block, return_stats=True,
        )
        ra = dflash_generate_adaptive(
            draft, target=target, input_ids=ids, max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id], temperature=args.temperature,
            block_size=base_block, max_block=max_block,
            adaptive_block=not args.no_adaptive_block, early_stop=not args.no_early_stop,
            min_block=args.min_block, conf_threshold=args.conf_threshold,
            ema_beta=args.ema_beta, block_factor=args.block_factor,
        )

        # losslessness check (greedy): both must equal target's own decode
        if args.temperature == 0.0:
            a = rf.output_ids[0, rf.num_input_tokens:].tolist()
            b = ra.output_ids[0, ra.num_input_tokens:].tolist()
            n = min(len(a), len(b))
            compared += 1
            if a[:n] != b[:n]:
                mismatches += 1
                first = next((k for k in range(n) if a[k] != b[k]), n)
                print(f"  [WARN] sample {i}: token mismatch at pos {first} "
                      f"(fixed_len={len(a)} adapt_len={len(b)})")

        fixed["tpot"].append(rf.time_per_output_token)
        fixed["accept"].append(np.mean(rf.acceptance_lengths))
        fixed["nfe"] += len(rf.acceptance_lengths)
        fixed["out_tok"] += rf.num_output_tokens
        fixed["blocks"].append(base_block)

        adapt["tpot"].append(ra.time_per_output_token)
        adapt["accept"].append(np.mean(ra.acceptance_lengths))
        adapt["nfe"] += ra.num_target_forwards
        adapt["out_tok"] += ra.num_output_tokens
        adapt["blocks"].extend(ra.block_sizes)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(dataset)}] "
                  f"fixed {1/np.mean(fixed['tpot']):.1f} tok/s acc={np.mean(fixed['accept']):.2f} | "
                  f"adapt {1/np.mean(adapt['tpot']):.1f} tok/s acc={np.mean(adapt['accept']):.2f} "
                  f"avgblk={np.mean(adapt['blocks']):.1f}")

    f_tpot = float(np.mean(fixed["tpot"]))
    a_tpot = float(np.mean(adapt["tpot"]))
    print("\n" + "=" * 64)
    print(f"{'metric':<28}{'fixed-block':>16}{'adaptive':>16}")
    print("-" * 64)
    print(f"{'throughput (tok/s)':<28}{1/f_tpot:>16.2f}{1/a_tpot:>16.2f}")
    print(f"{'mean acceptance length':<28}{np.mean(fixed['accept']):>16.2f}{np.mean(adapt['accept']):>16.2f}")
    print(f"{'target forwards (NFE)':<28}{fixed['nfe']:>16d}{adapt['nfe']:>16d}")
    print(f"{'output tokens (total)':<28}{fixed['out_tok']:>16d}{adapt['out_tok']:>16d}")
    print(f"{'tokens / target-forward':<28}"
          f"{fixed['out_tok']/max(fixed['nfe'],1):>16.2f}{adapt['out_tok']/max(adapt['nfe'],1):>16.2f}")
    print(f"{'avg drafted block':<28}{np.mean(fixed['blocks']):>16.2f}{np.mean(adapt['blocks']):>16.2f}")
    print("-" * 64)
    print(f"decode speedup (adaptive / fixed): {f_tpot / a_tpot:.3f}x")
    if args.temperature == 0.0:
        print(f"losslessness: {compared - mismatches}/{compared} samples token-identical to fixed-block DFlash")
    print("=" * 64)


if __name__ == "__main__":
    main()
