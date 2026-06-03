"""Adaptive-Block Confidence-Guided DFlash drafting (research prototype).

This extends the reference ``dflash_generate`` with two *lossless* efficiency
mechanisms, each independently toggleable for ablation:

1. Adaptive block size
   The number of tokens drafted per iteration is no longer fixed. A controller
   tracks an EMA of the realized acceptance length and sets the next block size
   proportional to it (clamped to ``[min_block, max_block]``). Easy / templated
   regions (high acceptance) get larger blocks (more parallelism); hard regions
   (low acceptance) shrink the block (less wasted draft+verify compute).

2. Confidence-guided early-stop
   The draft model already emits per-position logits in a single forward. We
   read the per-token confidence (max softmax prob) and truncate the drafted
   tail at the first token whose confidence falls below ``conf_threshold``
   (keeping that one token so the target can still correct it). Only the
   high-confidence prefix is sent to the target for verification, shortening the
   verify forward without hurting acceptance (low-confidence tail tokens are the
   ones most likely to be rejected anyway).

Losslessness: exactly as in the reference implementation, every committed token
is the target model's own ``posterior`` sample (accepted longest matching prefix
plus one bonus token). Neither mechanism changes which tokens are committed for a
given target distribution, so greedy output is identical to fixed-block DFlash.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache

from .model import extract_context_feature, sample


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate_adaptive(
    model,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    # --- adaptive controls ---
    adaptive_block: bool = True,
    early_stop: bool = True,
    min_block: int = 4,
    max_block: Optional[int] = None,
    conf_threshold: float = 0.6,
    ema_beta: float = 0.7,
    block_factor: float = 1.5,
    return_stats: bool = True,
):
    """Adaptive variant of ``dflash_generate``.

    Set ``adaptive_block=False, early_stop=False`` to recover the exact
    fixed-block reference behavior (useful as an in-harness sanity check).
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    base_block = model.block_size if block_size is None else block_size
    max_block = base_block if max_block is None else max_block
    max_block = max(max_block, base_block)
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    # Buffer must accommodate the largest possible block we might draft.
    output_ids = torch.full(
        (1, max_length + max_block), mask_token_id, dtype=torch.long, device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=base_block > 1,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    if base_block > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths: list[int] = []
    block_sizes: list[int] = []
    num_target_forwards = 0
    num_draft_forwards = 0
    verify_tokens = 0   # sum of eff_block: proxy for verify FLOPs (compute-bound regime)
    draft_tokens = 0    # sum of drafted positions: proxy for draft FLOPs
    start = num_input_tokens
    draft_prefill = True
    cur_block = base_block
    ema_accept = float(base_block)

    while start < max_length:
        cb = cur_block
        block_output_ids = output_ids[:, start : start + cb].clone()
        block_position_ids = position_ids[:, start : start + cb]

        eff_block = cb
        if base_block > 1 and cb > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + cb],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - cb :, :])
            past_key_values_draft.crop(start)
            num_draft_forwards += 1
            draft_tokens += cb - 1
            block_output_ids[:, 1:] = sample(draft_logits)

            if early_stop:
                # Confidence of each drafted token (max softmax prob).
                conf = torch.softmax(draft_logits.float(), dim=-1).max(dim=-1).values[0]
                below = (conf < conf_threshold).nonzero()
                if below.numel() > 0:
                    # keep the first low-confidence token so target can correct it
                    eff_draft = min(int(below[0].item()) + 1, cb - 1)
                else:
                    eff_draft = cb - 1
                eff_block = max(2, eff_draft + 1)

            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        ver_ids = block_output_ids[:, :eff_block]
        ver_pos = block_position_ids[:, :eff_block]
        output = target(
            ver_ids,
            position_ids=ver_pos,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=base_block > 1,
        )
        num_target_forwards += 1
        verify_tokens += eff_block

        posterior = sample(output.logits, temperature)
        acceptance_length = (ver_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = ver_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        committed = acceptance_length + 1
        start += committed
        past_key_values_target.crop(start)
        acceptance_lengths.append(committed)
        block_sizes.append(cb)

        if base_block > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :committed, :]

        # --- adaptive block-size controller ---
        if adaptive_block:
            ema_accept = ema_beta * ema_accept + (1.0 - ema_beta) * committed
            cur_block = int(round(ema_accept * block_factor))
            cur_block = max(min_block, min(max_block, cur_block))

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_t = torch.tensor(stop_token_ids, device=output_ids.device)
        idx = torch.isin(output_ids[0][num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + idx[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        block_sizes=block_sizes,
        num_target_forwards=num_target_forwards,
        num_draft_forwards=num_draft_forwards,
        verify_tokens=verify_tokens,
        draft_tokens=draft_tokens,
    )
