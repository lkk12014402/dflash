"""Probe whether a larger trained block (e.g. 32) + early-stop would help,
using the EXISTING block=16 draft (no training required).

Collects, over greedy decoding on gsm8k:
  (A) acceptance-length distribution at block=16 and block=32 (current draft)
      -> headroom: how often does acceptance approach / hit the ceiling?
  (B) per-position draft match rate + mean confidence (positions 1..K)
      -> does draft quality survive past position 16? (headroom for B=32)
  (C) confidence -> empirical accept calibration
      -> does early-stop's signal actually identify tokens to drop?
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from dflash.model import DFlashDraftModel, extract_context_feature, sample
from dflash.benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template


def attn_impl():
    try:
        import flash_attn  # noqa
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


@torch.inference_mode()
def run(model, target, input_ids, max_new_tokens, block_size, eos_id, rec):
    """Greedy DFlash loop (temp=0) that logs per-position (pos, conf, match)
    and per-iteration acceptance_length into `rec`."""
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    mask_id = model.mask_token_id
    out = torch.full((1, max_len + block_size), mask_id, dtype=torch.long, device=target.device)
    pos = torch.arange(out.shape[1], device=target.device).unsqueeze(0)
    pkv_t, pkv_d = DynamicCache(), DynamicCache()
    o = target(input_ids, position_ids=pos[:, :n_in], past_key_values=pkv_t,
               use_cache=True, logits_to_keep=1, output_hidden_states=True)
    out[:, :n_in] = input_ids
    out[:, n_in:n_in + 1] = sample(o.logits, 0.0)
    th = extract_context_feature(o.hidden_states, model.target_layer_ids)
    start = n_in
    while start < max_len:
        blk = out[:, start:start + block_size].clone()
        bpos = pos[:, start:start + block_size]
        ne = target.model.embed_tokens(blk)
        dl = target.lm_head(model(target_hidden=th, noise_embedding=ne,
                                  position_ids=pos[:, pkv_d.get_seq_length(): start + block_size],
                                  past_key_values=pkv_d, use_cache=True, is_causal=False)[:, 1 - block_size:, :])
        pkv_d.crop(start)
        conf = torch.softmax(dl.float(), dim=-1).max(dim=-1).values[0]  # [K]
        blk[:, 1:] = sample(dl)
        o = target(blk, position_ids=bpos, past_key_values=pkv_t, use_cache=True, output_hidden_states=True)
        post = sample(o.logits, 0.0)  # [1, block]
        match = (blk[:, 1:] == post[:, :-1])[0]      # raw per-position match (not prefix-gated)
        accept = match.float().cumprod(0).sum().int().item()  # prefix-gated acceptance length
        rec["accept"].append(accept)
        for i in range(block_size - 1):
            rec["pos"].append(i + 1)
            rec["conf"].append(float(conf[i]))
            rec["match"].append(bool(match[i].item()))
            rec["gated"].append(bool(match[:i + 1].all().item()))
        committed = accept + 1
        out[:, start:start + accept + 1] = blk[:, :accept + 1]
        out[:, start + accept + 1] = post[:, accept]
        start += committed
        pkv_t.crop(start)
        th = extract_context_feature(o.hidden_states, model.target_layer_ids)[:, :committed, :]
        if eos_id in out[0, n_in:]:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", required=True)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dataset", default="gsm8k")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    ai = attn_impl()
    t = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation=ai, dtype=torch.bfloat16).to(dev).eval()
    d = DFlashDraftModel.from_pretrained(args.draft_model, attn_implementation=ai, dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples)

    def ids(u):
        return tok.encode(_apply_chat_template(tok, [{"role": "user", "content": u}], False), return_tensors="pt").to(dev)

    for B in [16, 32]:
        rec = {"accept": [], "pos": [], "conf": [], "match": [], "gated": []}
        for inst in ds:
            run(d, t, ids(inst["turns"][0]), args.max_new_tokens, B, tok.eos_token_id, rec)
        acc = np.array(rec["accept"])
        pos = np.array(rec["pos"]); match = np.array(rec["match"]); gated = np.array(rec["gated"]); conf = np.array(rec["conf"])
        print(f"\n================ trained-block proxy: running current B16 draft at block={B} ================")
        print(f"(A) acceptance length: mean={acc.mean():.2f}  p50={np.percentile(acc,50):.0f}  "
              f"p90={np.percentile(acc,90):.0f}  max={acc.max()}  | ceiling={B-1}  "
              f"| %hit>=12: {100*(acc>=12).mean():.1f}%  %hit>=16: {100*(acc>=16).mean():.1f}%")
        print("(B) per-position GATED acceptance prob (prob the block reaches >= this position):")
        for lo in range(1, B, 4):
            sel = (pos >= lo) & (pos < lo + 4)
            if sel.any():
                print(f"    pos {lo:>2}-{lo+3:>2}:  reach-prob={gated[sel].mean():.3f}  "
                      f"raw-match={match[sel].mean():.3f}  mean-conf={conf[sel].mean():.3f}")
        # (C) calibration: empirical raw-match rate by confidence bin
        print("(C) confidence -> empirical raw-match (early-stop signal quality):")
        for lo, hi in [(0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,0.99),(0.99,1.01)]:
            sel = (conf >= lo) & (conf < hi)
            if sel.any():
                print(f"    conf [{lo:.2f},{hi:.2f}): n={sel.sum():>6}  P(match)={match[sel].mean():.3f}  "
                      f"P(gated-accept)={gated[sel].mean():.3f}")


if __name__ == "__main__":
    main()
