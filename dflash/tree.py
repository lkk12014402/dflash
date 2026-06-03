"""Diffusion tree-drafting + tree verification for DFlash (Innovation 1).

Motivation
----------
The reference DFlash draft computes a full per-position distribution for every
block position in ONE bidirectional forward, then **throws almost all of it
away**: ``block_output_ids[:, 1:] = sample(draft_logits)`` keeps only the
arg-max at each position. Those discarded top-k alternatives are exactly the
hedge a speculative decoder wants at the *acceptance frontier* (the first
position the greedy draft gets wrong).

Because the single-sequence target verify forward is **flat-cost** (memory-bound:
L=1..32 ~ 56-60ms), we can verify *many* candidate continuations in ONE target
forward for nearly the same latency as verifying one. So we:

1. read the draft's per-position top-k (free; already computed),
2. assemble them into a probability-ordered **token tree** (Sequoia/Medusa
   style; DFlash's per-position logits play the role of Medusa heads),
3. verify the whole tree in ONE target forward using a **tree-attention mask**
   (each node attends only to its ancestors + the committed prefix),
4. accept the longest tree path the target greedily agrees with, + 1 bonus
   token, and compact the accepted path's KV back into the cache.

Losslessness: committed tokens are the target's own greedy continuation along
the accepted path (identical to running the target autoregressively), so output
matches vanilla decoding -- modulo the same bf16 tie-breaking noise that already
affects the reference linear decoder.

``tree_max_nodes`` and ``top_k`` control the compute/acceptance trade-off.
Setting ``top_k=1`` collapses the tree to the reference linear chain.
"""

from __future__ import annotations

import heapq
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


def _build_tree(topk_ids, topk_p, max_nodes, max_depth):
    """Greedy probability-ordered tree over the draft's per-position top-k.

    ``topk_ids`` / ``topk_p`` : [depth, k] arg-sorted candidates for positions
    1..B-1 (depth d -> draft position d). Node 0 is the (already confirmed)
    root at depth 0. Returns flat node arrays (lists):
        tokens[i], depth[i], parent[i]  (parent[0] = -1)
    Expansion pops the globally highest path-probability frontier candidate
    (product of per-position marginals under the draft's independence), so the
    most likely continuations get the budget first.
    """
    tokens = [0]          # root token filled by caller
    depth = [0]
    parent = [-1]
    # frontier heap of (-score, tie, parent_node_idx, cand_depth, cand_rank)
    heap = []
    tie = 0
    k = topk_ids.shape[1]
    for r in range(k):
        heapq.heappush(heap, (-float(topk_p[0, r]), tie, 0, 1, r))
        tie += 1
    while heap and len(tokens) < max_nodes:
        neg_score, _, par, d, rank = heapq.heappop(heap)
        score = -neg_score
        node_idx = len(tokens)
        tokens.append(int(topk_ids[d - 1, rank]))
        depth.append(d)
        parent.append(par)
        if d < max_depth:
            for r in range(k):
                child_score = score * float(topk_p[d, r])
                heapq.heappush(heap, (-child_score, tie, node_idx, d + 1, r))
                tie += 1
    return tokens, depth, parent


@torch.inference_mode()
def dflash_generate_tree(
    model,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    # --- tree controls ---
    tree_max_nodes: int = 48,
    top_k: int = 3,
    return_stats: bool = True,
):
    """DFlash generation with diffusion tree-drafting + tree verification.

    ``top_k=1`` reduces the tree to the reference linear chain.
    """
    assert temperature < 1e-5, "tree verification prototype supports greedy (temperature=0) only"
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id
    max_depth = block_size - 1  # draft predicts positions 1..block_size-1

    buf = max_length + block_size + tree_max_nodes + 1
    output_ids = torch.full((1, buf), mask_token_id, dtype=torch.long, device=target.device)
    position_ids = torch.arange(buf, device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    neg_inf = torch.finfo(target.dtype).min

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths: list[int] = []
    tree_sizes: list[int] = []
    num_target_forwards = 0
    num_draft_forwards = 0
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        # ---- 1. draft forward: per-position distributions (one-shot) ----
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_output_ids[:, 1:] = mask_token_id
        draft_ctx_len = past_key_values_draft.get_seq_length()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, draft_ctx_len: start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size:, :])
        past_key_values_draft.crop(start)
        num_draft_forwards += 1

        probs = torch.softmax(draft_logits.float()[0], dim=-1)   # [B-1, V]
        topk_p, topk_ids = probs.topk(top_k, dim=-1)             # [B-1, k]
        topk_p = topk_p.cpu()
        topk_ids = topk_ids.cpu()

        # ---- 2. build probability-ordered tree ----
        tokens, depth, parent = _build_tree(topk_ids, topk_p, tree_max_nodes, max_depth)
        tokens[0] = int(block_output_ids[0, 0])
        N = len(tokens)
        tree_sizes.append(N)

        tok_t = torch.tensor([tokens], device=target.device)
        depth_t = torch.tensor(depth, device=target.device)
        pos_t = (start + depth_t).unsqueeze(0)
        cache_pos = torch.arange(start, start + N, device=target.device)

        # ancestor-or-self mask [N, N]
        anc = torch.zeros((N, N), dtype=torch.bool)
        for i in range(N):
            j = i
            while j != -1:
                anc[i, j] = True
                j = parent[j]
        mask = torch.full((1, 1, N, start + N), neg_inf, dtype=target.dtype, device=target.device)
        mask[..., :start] = 0
        tree_mask = torch.where(anc, torch.tensor(0.0), torch.tensor(neg_inf)).to(target.dtype)
        mask[0, 0, :, start:] = tree_mask.to(target.device)

        if draft_prefill and return_stats:
            draft_prefill = False
            decode_start = _cuda_time()

        # ---- 3. verify whole tree in one target forward ----
        output = target(
            tok_t,
            attention_mask=mask,
            position_ids=pos_t,
            past_key_values=past_key_values_target,
            use_cache=True,
            cache_position=cache_pos,
            output_hidden_states=True,
        )
        num_target_forwards += 1
        tgt_argmax = output.logits[0].argmax(dim=-1)   # [N] next-token per node

        # ---- 4. greedy tree verification: longest agreed path ----
        children = [[] for _ in range(N)]
        for i in range(1, N):
            children[parent[i]].append(i)
        accepted = [0]
        node = 0
        while True:
            want = int(tgt_argmax[node])
            nxt = None
            for c in children[node]:
                if tokens[c] == want:
                    nxt = c
                    break
            if nxt is None:
                break
            accepted.append(nxt)
            node = nxt
        bonus = int(tgt_argmax[node])
        m = len(accepted) - 1   # tree edges accepted

        # commit accepted path tokens + bonus
        for k_i in range(1, len(accepted)):
            output_ids[0, start + k_i] = tokens[accepted[k_i]]
        output_ids[0, start + m + 1] = bonus
        committed = m + 1
        acceptance_lengths.append(committed)

        # ---- 5. compact accepted KV back into the cache ----
        acc_t = torch.tensor(accepted, device=target.device)
        gather_idx = torch.cat([torch.arange(start, device=target.device), start + acc_t])
        for layer in past_key_values_target.layers:
            layer.keys = layer.keys[:, :, gather_idx, :].contiguous()
            layer.values = layer.values[:, :, gather_idx, :].contiguous()
        start += committed

        # update target_hidden from accepted path hidden states
        sel = [h[:, acc_t, :] for h in output.hidden_states]
        target_hidden = extract_context_feature(sel, model.target_layer_ids)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:start + 1] for stop_token_id in stop_token_ids
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
        tree_sizes=tree_sizes,
        num_target_forwards=num_target_forwards,
        num_draft_forwards=num_draft_forwards,
    )
