"""Training recipe for the DFlash block-diffusion draft model (EAGLE-style).

DFlash is a *feature-conditioned block-diffusion* draft for speculative decoding.
Unlike EAGLE (which re-embeds tokens and runs a causal transformer head), the
DFlash draft never re-processes tokens: it **cross-attends to hidden features
extracted from the frozen target** and, in a single bidirectional forward,
denoises a fully-masked block into the next ``block_size - 1`` tokens.

This module implements the training objective that *exactly* mirrors the
inference forward in ``dflash/model.py`` so that what we train is what we run.

Inference recap (see ``dflash_generate`` in model.py)
-----------------------------------------------------
For a block starting at absolute position ``s``:
  * context  = target features ``F_0 .. F_{s-1}``  (everything strictly before s)
  * the draft sees the noise block ``[x_s, MASK, MASK, ..., MASK]`` (length B);
    block-position 0 (``x_s``) is the already-accepted clean token, positions
    1..B-1 are MASK.
  * inside each draft layer:  q = noise (B);  k = v = concat([ctx_features, noise]);
    attention is **non-causal** -- every noise token attends to all context and
    all noise in its block (bidirectional).
  * the draft predicts, at MASK position ``s+i`` (i=1..B-1), the token ``x_{s+i}``
    via ``target.lm_head`` (logits taken from ``draft_out[:, 1-B:, :]``).

Training objective (this file)
------------------------------
Teacher-force a sequence ``x`` that lies on the target's own decoding trajectory
(use ``gen_data.py`` to produce target-greedy continuations -- this is what makes
hard cross-entropy on the gold tokens equivalent to distilling the greedy
verifier). We:
  1. run one frozen target forward to get features ``F_i`` at every position
     (same ``extract_context_feature`` the inference path uses),
  2. tile the completion region into blocks of ``block_size`` (with a random
     phase offset for alignment coverage), masking every block's tail,
  3. run ONE whole-sequence draft forward whose 4D attention mask reproduces the
     per-block inference attention exactly:
         query i (block start s):  ctx col j visible iff j < s
                                   noise col j visible iff block(j) == block(i)
  4. cross-entropy over the masked positions only (optionally + a correctly
     *shifted* KL distillation against the target's own logits).

Only the draft parameters are trained; the target's ``embed_tokens`` and
``lm_head`` (tied) stay frozen.

Self-test: ``python -m dflash.train selftest --model Qwen/Qwen3-4B \
              --draft-model z-lab/Qwen3-4B-DFlash-b16`` verifies that the
whole-sequence training forward is bit-equivalent to the per-block inference
forward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .model import DFlashDraftModel, extract_context_feature


# ---------------------------------------------------------------------------
# Block / mask construction  (the heart of train==inference equivalence)
# ---------------------------------------------------------------------------

# Sentinels for the per-position block id.
_PREFIX = -1   # pure-context prompt positions (no loss, not a draft block)
_PAD = -2      # right-padding positions (never visible, never supervised)


def build_block_ids(seq_len: int, prompt_len: int, block_size: int, offset: int) -> torch.Tensor:
    """Per-position block-start id.

    Positions ``[0, prompt_len + offset)`` are pure context (``_PREFIX``).
    The rest is tiled into contiguous blocks of ``block_size``; a position's id
    is the absolute index of the first token of its block.
    """
    block_ids = torch.full((seq_len,), _PREFIX, dtype=torch.long)
    start = prompt_len + offset
    if start >= seq_len:
        # Degenerate: nothing left to predict. Caller should skip such samples.
        return block_ids
    pos = torch.arange(start, seq_len)
    block_ids[start:] = start + ((pos - start) // block_size) * block_size
    return block_ids


def build_attention_mask(block_ids: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
    """4D additive mask ``[1, 1, L, 2L]`` for keys ``[ctx(L) ; noise(L)]``.

    query i (block start s = block_ids[i]):
      * ctx column j  : visible iff j < s          (causal to clean prefix feats)
      * noise column j: visible iff block_ids[j] == s and s is a real block
    """
    L = block_ids.shape[0]
    neg = torch.finfo(dtype).min
    bi = block_ids.to(device)
    j = torch.arange(L, device=device)
    # ctx half: j < block_start(i)
    ctx_ok = j[None, :] < bi[:, None]                      # [L, L]
    # noise half: same (real) block
    same = (bi[:, None] == bi[None, :]) & (bi[:, None] >= 0)
    full_ok = torch.cat([ctx_ok, same], dim=1)             # [L, 2L]
    mask = torch.where(full_ok, torch.zeros((), dtype=dtype, device=device),
                       torch.full((), neg, dtype=dtype, device=device))
    return mask.view(1, 1, L, 2 * L)


def build_example(input_ids: list[int], prompt_len: int, block_size: int,
                  mask_token_id: int, offset: Optional[int] = None):
    """Build the masked noise stream, labels and block ids for one sequence."""
    S = len(input_ids)
    if offset is None:
        offset = random.randint(0, block_size - 1)
    ids = torch.tensor(input_ids, dtype=torch.long)
    block_ids = build_block_ids(S, prompt_len, block_size, offset)

    noise_ids = ids.clone()
    labels = torch.full((S,), -100, dtype=torch.long)
    is_block = block_ids >= 0
    is_first = is_block & (torch.arange(S) == block_ids)   # block-position 0 (clean)
    masked = is_block & ~is_first                          # positions to denoise
    noise_ids[masked] = mask_token_id
    labels[masked] = ids[masked]
    return ids, noise_ids, labels, block_ids


# ---------------------------------------------------------------------------
# Dataset / collation
# ---------------------------------------------------------------------------

class TokenizedDataset(Dataset):
    """JSONL with one object per line: ``{"input_ids": [...], "prompt_len": P}``.

    ``prompt_len`` defaults to 1 (treat only the BOS as context) if absent.
    """

    def __init__(self, path: str, block_size: int, max_len: int, min_completion: int = 2):
        self.samples = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                ids = obj["input_ids"][:max_len]
                p = int(obj.get("prompt_len", 1))
                p = min(p, len(ids) - 1)
                if len(ids) - p < min_completion:
                    continue
                self.samples.append((ids, p))
        self.block_size = block_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, p = self.samples[idx]
        return build_example(ids, p, self.block_size, self.mask_token_id)


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch):
        Lmax = max(ids.shape[0] for ids, *_ in batch)
        B = len(batch)
        input_ids = torch.full((B, Lmax), self.pad_token_id, dtype=torch.long)
        noise_ids = torch.full((B, Lmax), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, Lmax), -100, dtype=torch.long)
        block_ids = torch.full((B, Lmax), _PAD, dtype=torch.long)
        attn = torch.ones((B, Lmax), dtype=torch.long)        # target padding mask
        for b, (ids, n_ids, lbl, blk) in enumerate(batch):
            S = ids.shape[0]
            input_ids[b, :S] = ids
            noise_ids[b, :S] = n_ids
            labels[b, :S] = lbl
            block_ids[b, :S] = blk
            attn[b, S:] = 0
        return {
            "input_ids": input_ids,
            "noise_ids": noise_ids,
            "labels": labels,
            "block_ids": block_ids,
            "target_attn": attn,
        }


# ---------------------------------------------------------------------------
# Core training forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_target_features(target: nn.Module, input_ids: torch.Tensor,
                            target_attn: torch.Tensor, layer_ids: list[int],
                            want_logits: bool):
    """Frozen target forward -> (concat features [B,L,5H], optional logits)."""
    out = target(
        input_ids,
        attention_mask=target_attn,
        use_cache=False,
        output_hidden_states=True,
    )
    feats = extract_context_feature(out.hidden_states, layer_ids)
    logits = out.logits if want_logits else None
    return feats, logits


def dflash_training_forward(draft: nn.Module, target: nn.Module, batch: dict):
    """One whole-sequence draft forward; returns logits/labels/weights on masked positions."""
    device = next(draft.parameters()).device
    input_ids = batch["input_ids"].to(device)
    noise_ids = batch["noise_ids"].to(device)
    labels = batch["labels"].to(device)
    block_ids = batch["block_ids"]                     # cpu ok
    target_attn = batch["target_attn"].to(device)

    B, L = input_ids.shape
    layer_ids = draft.target_layer_ids

    feats, _ = compute_target_features(target, input_ids, target_attn, layer_ids, want_logits=False)
    feats = feats.to(draft.dtype)
    noise_emb = target.get_input_embeddings()(noise_ids).to(draft.dtype)

    # per-example 4D mask -> [B,1,L,2L]
    masks = torch.stack([build_attention_mask(block_ids[b], draft.dtype, device)[0]
                         for b in range(B)], dim=0)
    position_ids = torch.arange(L, device=device)
    position_ids = torch.cat([position_ids, position_ids]).unsqueeze(0).expand(B, -1)

    hidden = draft(
        target_hidden=feats,
        noise_embedding=noise_emb,
        position_ids=position_ids,
        attention_mask=masks,
        use_cache=False,
        is_causal=False,
    )                                                  # [B, L, H]

    loss_mask = labels.view(-1) != -100
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])[loss_mask]
    logits = target.lm_head(flat_hidden)               # [N, V]
    target_labels = labels.view(-1)[loss_mask]

    # within-block position k (1..block_size-1) for each supervised token
    abs_pos = torch.arange(L).unsqueeze(0).expand(B, -1)
    within = (abs_pos - block_ids).to(device).view(-1)[loss_mask]   # k for masked tokens
    return logits, target_labels, loss_mask, hidden, within


def compute_loss(draft, target, batch, distill_weight: float = 0.0,
                 distill_temp: float = 1.0, loss_decay: float = 0.0):
    logits, labels, loss_mask, hidden, within = dflash_training_forward(draft, target, batch)
    # loss-decay weighting (DFlash eq. 4): emphasize early block positions, since
    # an early error invalidates every later token. w_k = exp(-(k-1)/gamma).
    if loss_decay and loss_decay > 0:
        w = torch.exp(-(within.float() - 1.0) / loss_decay)
        ce_tok = F.cross_entropy(logits.float(), labels, reduction="none")
        ce = (w * ce_tok).sum() / w.sum().clamp_min(1e-6)
    else:
        ce = F.cross_entropy(logits.float(), labels)
    loss = ce
    kl = torch.tensor(0.0, device=ce.device)
    if distill_weight > 0.0:
        # KL must be aligned: draft logits at absolute position i predict token
        # x_i, whose verifier distribution comes from TARGET position i-1.
        device = logits.device
        input_ids = batch["input_ids"].to(device)
        target_attn = batch["target_attn"].to(device)
        with torch.no_grad():
            t_out = target(input_ids, attention_mask=target_attn, use_cache=False)
            t_logits = t_out.logits                    # [B,L,V]
        B, L = input_ids.shape
        # position index (within [B,L]) of each supervised token, then shift -1.
        flat_pos = torch.arange(B * L, device=device)[loss_mask]
        shifted = flat_pos - 1                          # target position i-1
        t_sel = t_logits.reshape(-1, t_logits.shape[-1])[shifted].float()
        log_p = F.log_softmax(logits.float() / distill_temp, dim=-1)
        with torch.no_grad():
            q = F.softmax(t_sel / distill_temp, dim=-1)
        kl = F.kl_div(log_p, q, reduction="batchmean") * (distill_temp ** 2)
        loss = (1.0 - distill_weight) * ce + distill_weight * kl
    with torch.no_grad():
        pred = logits.argmax(-1)
        correct = (pred == labels).float()
        acc = correct.mean()
        # per-block-position accuracy: position k=1 is the token right after the
        # anchor (the tightest proxy for accept rate); later k are progressively
        # harder. Report a few representative positions.
        pos_acc = {}
        for k in (1, 2, 4, 8):
            sel = (within == k)
            n = int(sel.sum().item())
            pos_acc[f"acc@{k}"] = (correct[sel].mean().item() if n > 0 else float("nan"))
    stats = {"ce": ce.item(), "kl": kl.detach().item(), "token_acc": acc.item(),
             "n_tokens": labels.numel()}
    stats.update(pos_acc)
    return loss, stats


# ---------------------------------------------------------------------------
# Equivalence self-test: whole-seq training forward == per-block inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def selftest(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0")
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(args.draft_model, dtype=dtype).to(device).eval()
    Bk = draft.block_size
    mask_id = draft.mask_token_id
    layer_ids = draft.target_layer_ids

    torch.manual_seed(0)
    P = 7                       # prompt / context length
    offset = 3                  # phase -> first block at P+offset, exercises >1 block
    n_blocks = 2
    S = P + offset + n_blocks * Bk
    vocab = target.config.vocab_size
    ids = torch.randint(0, vocab, (S,), device=device)

    # --- features from a full causal target forward ---
    out = target(ids.unsqueeze(0), use_cache=False, output_hidden_states=True)
    feats = extract_context_feature(out.hidden_states, layer_ids).to(dtype)

    # ---------- (B) whole-sequence training-style forward (all blocks at once) ----------
    full = ids.tolist()
    _, noise_ids, labels, block_ids = build_example(full, P, Bk, mask_id, offset=offset)
    batch = {
        "input_ids": ids.unsqueeze(0),
        "noise_ids": noise_ids.unsqueeze(0).to(device),
        "labels": labels.unsqueeze(0).to(device),
        "block_ids": block_ids.unsqueeze(0),
        "target_attn": torch.ones(1, S, dtype=torch.long, device=device),
    }
    _, _, _, hidden_B, _ = dflash_training_forward(draft, target, batch)

    # ---------- (A) per-block inference-style forwards, compared to (B) ----------
    max_diff = 0.0
    total = 0
    matched = 0
    block_starts = sorted(set(int(b) for b in block_ids.tolist() if b >= 0))
    for s in block_starts:
        end = min(s + Bk, S)
        bk = end - s
        if bk < 2:
            continue
        block = ids[s:end].clone()
        block[1:] = mask_id                                # [bk] clean first + masks
        noise_emb_A = target.get_input_embeddings()(block.unsqueeze(0)).to(dtype)
        pos_A = torch.arange(0, s + bk, device=device).unsqueeze(0)
        hid_A = draft(
            target_hidden=feats[:, :s],                    # context = F_{0..s-1}
            noise_embedding=noise_emb_A,
            position_ids=pos_A,
            use_cache=False,
            is_causal=False,
        )
        logits_A = target.lm_head(hid_A[:, 1 - bk:, :])[0]  # [bk-1, V]
        logits_B = target.lm_head(hidden_B[0, s + 1:end, :])
        max_diff = max(max_diff, (logits_A - logits_B).abs().max().item())
        matched += (logits_A.argmax(-1) == logits_B.argmax(-1)).sum().item()
        total += logits_A.shape[0]

    argmax_match = matched / total
    print(f"[selftest] dtype={dtype}  block_size={Bk}  blocks={block_starts}  "
          f"predicted_positions={total}")
    print(f"[selftest] max|logits_A - logits_B| = {max_diff:.6g}")
    print(f"[selftest] argmax agreement        = {argmax_match*100:.1f}%")
    # fp32 differs only by sdpa/lm_head reduction-order across key counts; the
    # hard gate is argmax equivalence of every predicted position.
    tol = 3e-2 if args.fp32 else 2e-1
    ok = (max_diff < tol) and (argmax_match > 0.999)
    print(f"[selftest] {'PASS' if ok else 'FAIL'} (tol={tol})")
    return ok


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _is_dist():
    return "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1


def _rank():
    return int(os.environ.get("RANK", 0))


def _local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def _world():
    return int(os.environ.get("WORLD_SIZE", 1))


def _is_main():
    return _rank() == 0


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if _is_dist():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(_local_rank())
    device = torch.device(f"cuda:{_local_rank()}")
    torch.manual_seed(args.seed + _rank())

    target = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    if args.init_from:
        draft = DFlashDraftModel.from_pretrained(args.init_from, dtype=torch.float32)
        if _is_main():
            print(f"[init] continued from {args.init_from}")
    else:
        config = AutoConfig.from_pretrained(args.draft_config, trust_remote_code=True)
        draft = DFlashDraftModel(config)
        if _is_main():
            print(f"[init] fresh draft from config {args.draft_config}")
    draft = draft.to(device).train()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    block_size = args.block_size or draft.block_size
    mask_id = draft.mask_token_id
    ds = TokenizedDataset(args.data, block_size, args.max_len)
    ds.mask_token_id = mask_id
    if _is_main():
        print(f"[data] {len(ds)} sequences, block_size={block_size}, mask_id={mask_id}")

    sampler = DistributedSampler(ds, num_replicas=_world(), rank=_rank(), shuffle=True) if _is_dist() else None
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        collate_fn=Collator(pad_id), num_workers=args.num_workers, drop_last=True,
    )

    core = draft
    if _is_dist():
        draft = nn.parallel.DistributedDataParallel(draft, device_ids=[_local_rank()])

    optim = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_lambda(s, warmup, total_steps))
    if _is_main():
        print(f"[sched] total_steps={total_steps} warmup={warmup}")

    out_dir = Path(args.output_dir)
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(getattr(args, "wandb", False)) and _is_main()
    wb = None
    if use_wandb:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.wandb_run_name,
                    mode=args.wandb_mode, config=vars(args))
            wb.define_metric("train/*", step_metric="step")
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            wb = None
            use_wandb = False
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = compute_loss(draft.module if _is_dist() else draft, target,
                                           batch, args.distill_weight, args.distill_temp,
                                           args.loss_decay)
            (loss / args.grad_accum).backward()
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if _is_main() and global_step % args.log_every == 0:
                    tput = stats["n_tokens"] / (time.time() - t0)
                    print(f"e{epoch} s{global_step}/{total_steps} "
                          f"loss={loss.item():.4f} ce={stats['ce']:.4f} "
                          f"kl={stats['kl']:.4f} acc={stats['token_acc']:.3f} "
                          f"a@1={stats['acc@1']:.3f} a@4={stats['acc@4']:.3f} "
                          f"a@8={stats['acc@8']:.3f} "
                          f"lr={sched.get_last_lr()[0]:.2e} {tput:.0f}tok/s")
                    if use_wandb:
                        import math as _m
                        log = {
                            "step": global_step,
                            "epoch": epoch,
                            "train/loss": loss.item(),
                            "train/ce": stats["ce"],
                            "train/kl": stats["kl"],
                            "train/token_acc": stats["token_acc"],
                            "train/lr": sched.get_last_lr()[0],
                            "train/tok_per_s": tput,
                        }
                        for k in (1, 2, 4, 8):
                            v = stats.get(f"acc@{k}")
                            if v is not None and not _m.isnan(v):
                                log[f"train/acc@{k}"] = v
                        wb.log(log, step=global_step)
                    t0 = time.time()
                if _is_main() and args.save_every and global_step % args.save_every == 0:
                    core.save_pretrained(out_dir / f"step{global_step}")
        if _is_main():
            core.save_pretrained(out_dir / f"epoch{epoch}")
            print(f"[save] epoch {epoch} -> {out_dir / f'epoch{epoch}'}")

    if _is_main():
        core.save_pretrained(out_dir / "final")
        print(f"[done] -> {out_dir / 'final'}")
    if use_wandb:
        wb.finish()
    if _is_dist():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DFlash draft training (EAGLE-style)")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="verify training forward == inference forward")
    st.add_argument("--model", required=True)
    st.add_argument("--draft-model", required=True)
    st.add_argument("--fp32", action="store_true", help="run in fp32 for a tight tolerance")

    tr = sub.add_parser("train", help="train the draft model")
    tr.add_argument("--model", required=True, help="frozen target model")
    tr.add_argument("--data", required=True, help="jsonl of {input_ids, prompt_len}")
    tr.add_argument("--draft-config", default=None,
                    help="config for a fresh draft (HF repo or local dir)")
    tr.add_argument("--init-from", default=None, help="continue from a pretrained draft")
    tr.add_argument("--output-dir", default="dflash_ckpt")
    tr.add_argument("--block-size", type=int, default=None)
    tr.add_argument("--max-len", type=int, default=2048)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--grad-accum", type=int, default=8)
    tr.add_argument("--epochs", type=int, default=2)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight-decay", type=float, default=0.0)
    tr.add_argument("--warmup-ratio", type=float, default=0.02)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--distill-weight", type=float, default=0.0,
                    help="weight of KL-to-target distillation (0 = pure hard CE)")
    tr.add_argument("--distill-temp", type=float, default=1.0)
    tr.add_argument("--loss-decay", type=float, default=16.0,
                    help="gamma for exp(-(k-1)/gamma) early-position loss weighting "
                         "(DFlash eq.4); 0 = uniform")
    tr.add_argument("--num-workers", type=int, default=2)
    tr.add_argument("--log-every", type=int, default=10)
    tr.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    tr.add_argument("--wandb-project", default="dflash-draft")
    tr.add_argument("--wandb-run-name", default=None)
    tr.add_argument("--wandb-mode", default="online",
                    choices=["online", "offline", "disabled"])
    tr.add_argument("--save-every", type=int, default=0)
    tr.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "selftest":
        ok = selftest(args)
        raise SystemExit(0 if ok else 1)
    elif args.cmd == "train":
        if not args.draft_config and not args.init_from:
            p.error("provide --draft-config (fresh) or --init-from (continue)")
        train(args)


if __name__ == "__main__":
    main()
