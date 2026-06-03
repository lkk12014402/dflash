"""Iterative draft self-refinement for DFlash (research prototype, Innovation 2).

Background
----------
The reference ``dflash_generate`` fills a whole masked block in **one** draft
forward: every position 1..B-1 is predicted simultaneously while attending only
to position 0 (the confirmed token) and the target-hidden context. This is the
classic block-diffusion one-shot ("0 refinement steps") generation. Tokens deep
in the block are predicted with essentially no information about their immediate
neighbours, so their per-position confidence -- and their probability of being
accepted by the target -- collapses past the model's predictability horizon.

Innovation 2: cheap iterative self-refinement
----------------------------------------------
DFlash's draft is a *tiny* network (5 layers) and the target verify forward is
**flat-cost** in the single-sequence regime (memory-bound: L=1..32 ~ 56-60ms).
So throughput is governed by *acceptance per target forward*, not by verify
length, and a cheap draft-quality boost is almost free in tok/s.

We import the confidence-thresholded iterative unmasking used by Nemotron's
diffusion path into DFlash's one-shot draft:

  pass 0 : all of 1..B-1 masked -> draft forward -> freeze positions whose
           confidence >= ``refine_threshold`` (commit their sampled token).
  pass k : re-mask the still-unfrozen positions, run the draft forward again.
           Now the frozen high-confidence tokens are real embeddings, so the
           remaining masked positions get *bidirectional* context from their
           neighbours and predict more accurately.
  last   : fill every still-masked position from the latest sample (no re-mask).

Only the *draft proposal* changes; the target still verifies the whole block and
commits the longest matching prefix + one bonus token. Losslessness is therefore
identical to the reference: committed tokens are always the target's own sample.

``num_refine=0`` reproduces the reference one-shot draft exactly.
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
def dflash_generate_refine(
    model,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    # --- self-refinement controls ---
    num_refine: int = 1,
    refine_threshold: float = 0.6,
    return_stats: bool = True,
):
    """DFlash generation with iterative draft self-refinement.

    Set ``num_refine=0`` to recover the exact one-shot reference draft.
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=target.device,
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
        output_hidden_states=block_size > 1,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths: list[int] = []
    num_target_forwards = 0
    num_draft_forwards = 0
    start = num_input_tokens
    draft_prefill = True

    def _draft_forward(block_output_ids, draft_ctx_len):
        """One draft pass over the current block; returns per-position logits.

        The draft KV cache is restored to ``draft_ctx_len`` (its state *before*
        this block was drafted) so every refinement pass sees identical shapes:
        position_ids span [draft_ctx_len : start+block_size] == ctx_len+block.
        """
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, draft_ctx_len: start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size:, :])
        return draft_logits

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        if block_size > 1:
            # positions 1.. start masked; position 0 is the confirmed token.
            block_output_ids[:, 1:] = mask_token_id
            draft_ctx_len = past_key_values_draft.get_seq_length()

            for it in range(num_refine + 1):
                if it > 0:
                    # restore cache to pre-block state before re-running.
                    past_key_values_draft.crop(draft_ctx_len)
                draft_logits = _draft_forward(block_output_ids, draft_ctx_len)
                num_draft_forwards += 1

                samp = sample(draft_logits)[0]                         # [block-1]
                conf = torch.softmax(draft_logits.float(), dim=-1).max(dim=-1).values[0]
                row = block_output_ids[0, 1:]
                is_mask = row == mask_token_id
                last = it == num_refine
                if last:
                    new_row = torch.where(is_mask, samp, row)
                else:
                    freeze = is_mask & (conf >= refine_threshold)
                    new_row = torch.where(freeze, samp, row)
                block_output_ids[0, 1:] = new_row

            past_key_values_draft.crop(start)

            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )
        num_target_forwards += 1

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]

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
        num_target_forwards=num_target_forwards,
        num_draft_forwards=num_draft_forwards,
    )
