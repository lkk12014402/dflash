"""Ablation: fixed-block DFlash vs adaptive mechanisms, reporting BOTH
- memory-bound metric: throughput (tok/s)  [single-seq decode here]
- compute-bound proxy: verify-tokens & draft-tokens (FLOPs that scale with
  block length under batched / long-context serving)

so the regime-dependent tradeoff is explicit. Greedy => losslessness checked
against the fixed-block reference.

  python ablation_adaptive.py --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 --max-samples 16 --max-new-tokens 256
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.model import DFlashDraftModel, dflash_generate
from dflash.adaptive import dflash_generate_adaptive
from dflash.benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template


def attn_impl():
    try:
        import flash_attn  # noqa
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", required=True)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--max-samples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    ai = attn_impl()
    t = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation=ai, dtype=torch.bfloat16).to(dev).eval()
    d = DFlashDraftModel.from_pretrained(args.draft_model, attn_implementation=ai, dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    B = d.block_size
    ds = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples)

    def ids(u):
        return tok.encode(_apply_chat_template(tok, [{"role": "user", "content": u}], False), return_tensors="pt").to(dev)

    dflash_generate(d, target=t, input_ids=ids("Hi"), max_new_tokens=8,
                    stop_token_ids=[tok.eos_token_id], temperature=0.0, block_size=B, return_stats=True)

    configs = {
        f"fixed (B={B})":        dict(kind="fixed"),
        "early-stop@0.7":        dict(kind="adapt", adaptive_block=False, early_stop=True,  conf_threshold=0.7),
        "early-stop@0.5":        dict(kind="adapt", adaptive_block=False, early_stop=True,  conf_threshold=0.5),
        "adaptive-block":        dict(kind="adapt", adaptive_block=True,  early_stop=False, block_factor=1.5),
        "adaptive+earlystop":    dict(kind="adapt", adaptive_block=True,  early_stop=True,  conf_threshold=0.6, block_factor=1.5),
    }

    # reference (fixed) outputs for losslessness check
    ref_out = {}
    rows = {}
    for name, cfg in configs.items():
        tpot, acc = [], []
        nfe = vtok = dtok = otok = 0
        mism = 0
        for i, inst in enumerate(ds):
            x = ids(inst["turns"][0])
            if cfg["kind"] == "fixed":
                r = dflash_generate(d, target=t, input_ids=x, max_new_tokens=args.max_new_tokens,
                                    stop_token_ids=[tok.eos_token_id], temperature=0.0, block_size=B, return_stats=True)
                out = r.output_ids[0, r.num_input_tokens:].tolist()
                ref_out[i] = out
                nfe += len(r.acceptance_lengths)
                vtok += sum(r.acceptance_lengths)  # fixed verifies full block; approximate via committed+rejected
                # exact verify tokens for fixed = iterations * B (drafts full block each step)
                vtok = vtok  # placeholder; recompute below
                otok += r.num_output_tokens
                acc.append(np.mean(r.acceptance_lengths))
                tpot.append(r.time_per_output_token)
            else:
                r = dflash_generate_adaptive(d, target=t, input_ids=x, max_new_tokens=args.max_new_tokens,
                                             stop_token_ids=[tok.eos_token_id], temperature=0.0,
                                             block_size=B, max_block=B, min_block=4, **{k: v for k, v in cfg.items() if k != "kind"})
                out = r.output_ids[0, r.num_input_tokens:].tolist()
                nfe += r.num_target_forwards
                vtok += r.verify_tokens
                dtok += r.draft_tokens
                otok += r.num_output_tokens
                acc.append(np.mean(r.acceptance_lengths))
                tpot.append(r.time_per_output_token)
                ref = ref_out.get(i)
                if ref is not None:
                    n = min(len(ref), len(out))
                    if ref[:n] != out[:n]:
                        mism += 1
        if cfg["kind"] == "fixed":
            vtok = nfe * B            # fixed always verifies a full block
            dtok = nfe * (B - 1)
        rows[name] = dict(tps=1 / np.mean(tpot), acc=float(np.mean(acc)), nfe=nfe,
                          vtok=vtok, dtok=dtok, otok=otok, mism=mism, kind=cfg["kind"])

    base = rows[f"fixed (B={B})"]
    print("\n" + "=" * 96)
    print(f"{'config':<20}{'tok/s':>9}{'accept':>8}{'NFE':>7}{'verify_tok':>12}{'draft_tok':>11}"
          f"{'verify_saved':>13}{'lossless':>11}")
    print("-" * 96)
    for name, r in rows.items():
        vsav = 100.0 * (1 - r["vtok"] / base["vtok"]) if base["vtok"] else 0.0
        loss = "ref" if r["kind"] == "fixed" else f"{args.max_samples - r['mism']}/{args.max_samples}"
        print(f"{name:<20}{r['tps']:>9.1f}{r['acc']:>8.2f}{r['nfe']:>7d}{r['vtok']:>12d}{r['dtok']:>11d}"
              f"{vsav:>12.1f}%{loss:>11}")
    print("-" * 96)
    print("Reading the table:")
    print(" * tok/s  -> single-seq, memory-bound regime (verify forward ~flat in length): higher is better.")
    print(" * verify_tok / verify_saved -> compute-bound proxy (batched/long-ctx serving): lower verify_tok")
    print("   means fewer scaling FLOPs; early-stop trades a little acceptance for large verify savings.")
    print("=" * 96)


if __name__ == "__main__":
    main()
