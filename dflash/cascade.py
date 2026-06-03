"""Cascaded tree verification for DFlash (Innovation B).

The plain tree decoder (``dflash/tree.py``) builds the candidate tree from the
draft's per-position **marginal** top-k (each position scored independently,
with every *other* block position masked). Throughput peaks at ~64 nodes because
the target verify stops being flat beyond that. To spend the (scarce) verify
budget on *better* candidates we add a cheap **cascade**:

  1. marginal draft pass (cached, faithful) -> per-position top-k.
  2. build a LARGE candidate tree (``big_nodes``) from the marginals.
  3. PRUNER: one *cacheless* draft pass over the big tree with an **ancestor
     mask**, so every node sees its real ancestor tokens (not masks). The slot of
     each node then predicts a distribution *conditioned on the actual path*;
     score child ``i`` = log p_draft(token_i | ancestors of i). This is the
     refinement signal -- useless for a *linear* prefix verifier, but exactly
     cashable here because it re-ranks *competing candidates*.
  4. re-expand a small ``keep_nodes`` subtree using the conditional path scores
     (parent-before-child), and full-verify only that subtree on the target.

The pruner pass is ~5-layer draft over big_nodes tokens (cheap vs the target),
so we can shortlist from a much wider candidate pool than we could afford to
verify. Losslessness is unchanged: the target still verifies and commits the
longest agreed path + bonus token.

``prune=False`` (or ``keep_nodes>=big_nodes``) degrades to plain marginal tree.
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
from .tree import _build_tree, _cuda_time


def _reexpand(parent_big, child_score, keep_nodes):
    """Pick a connected size-``keep_nodes`` subtree of the big tree maximising
    cumulative conditional path score (parent always kept before child).

    ``child_score[i]`` = conditional log-prob of node i given its ancestors
    (``-inf`` semantics not needed; root has score 0). Returns the list of kept
    big-tree node indices (root first), and a remap old->new index.
    """
    n_big = len(parent_big)
    children = [[] for _ in range(n_big)]
    for i in range(1, n_big):
        children[parent_big[i]].append(i)

    kept = [0]
    in_kept = {0}
    # frontier heap of (-path_score, tie, node)
    heap = []
    tie = 0
    path_score = [0.0] * n_big
    for c in children[0]:
        path_score[c] = child_score[c]
        heapq.heappush(heap, (-path_score[c], tie, c)); tie += 1
    while heap and len(kept) < keep_nodes:
        neg, _, node = heapq.heappop(heap)
        kept.append(node)
        in_kept.add(node)
        for c in children[node]:
            path_score[c] = path_score[node] + child_score[c]
            heapq.heappush(heap, (-path_score[c], tie, c)); tie += 1

    kept.sort()  # keep ancestors before descendants (parent idx < child idx holds in build order)
    remap = {old: new for new, old in enumerate(kept)}
    return kept, remap


@torch.inference_mode()
def dflash_generate_cascade(
    model,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    # --- cascade controls ---
    big_nodes: int = 192,
    keep_nodes: int = 48,
    top_k: int = 4,
    prune: bool = True,
    return_stats: bool = True,
):
    """DFlash tree decoding with a cheap conditional-rescoring cascade.

    ``prune=False`` reproduces the plain marginal tree of ``dflash/tree.py``
    (with ``tree_max_nodes=keep_nodes``).
    """
    assert temperature < 1e-5, "cascade prototype is greedy-only (temperature=0)"
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id
    max_depth = block_size - 1

    buf = max_length + block_size + max(big_nodes, keep_nodes) + 1
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
    big_sizes: list[int] = []
    num_target_forwards = 0
    num_draft_forwards = 0
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        # ---- 1. marginal draft pass (cached, faithful) ----
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_output_ids[:, 1:] = mask_token_id
        draft_ctx_len = past_key_values_draft.get_seq_length()
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=target.model.embed_tokens(block_output_ids),
            position_ids=position_ids[:, draft_ctx_len: start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size:, :])
        past_key_values_draft.crop(start)
        num_draft_forwards += 1

        probs = torch.softmax(draft_logits.float()[0], dim=-1)
        topk_p, topk_ids = probs.topk(top_k, dim=-1)
        topk_p_c, topk_ids_c = topk_p.cpu(), topk_ids.cpu()

        # ---- 2. build big candidate tree ----
        nodes_big, depth_big, parent_big = _build_tree(topk_ids_c, topk_p_c, big_nodes, max_depth)
        nodes_big[0] = int(block_output_ids[0, 0])
        nb = len(nodes_big)
        big_sizes.append(nb)

        if prune and nb > keep_nodes:
            # ---- 3. PRUNER: cacheless conditional draft pass over big tree ----
            ctx_len = target_hidden.shape[1]
            depth_bt = torch.tensor(depth_big, device=target.device)
            ctx_pos = torch.arange(start - ctx_len, start, device=target.device)
            noise_pos = start + depth_bt
            pr_pos = torch.cat([ctx_pos, noise_pos]).unsqueeze(0)

            anc = torch.zeros((nb, nb), dtype=torch.bool)
            for i in range(nb):
                j = i
                while j != -1:
                    anc[i, j] = True
                    j = parent_big[j]
            pr_mask = torch.full((1, 1, nb, ctx_len + nb), neg_inf, dtype=target.dtype, device=target.device)
            pr_mask[..., :ctx_len] = 0
            pr_mask[0, 0, :, ctx_len:] = torch.where(
                anc.to(target.device), torch.tensor(0.0, device=target.device),
                torch.tensor(neg_inf, device=target.device),
            ).to(target.dtype)

            nodes_big_t = torch.tensor([nodes_big], device=target.device)
            pr_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=target.model.embed_tokens(nodes_big_t),
                position_ids=pr_pos,
                attention_mask=pr_mask,
                use_cache=False,
                is_causal=False,
            ))[0]   # [nb, V]
            num_draft_forwards += 1
            logp = torch.log_softmax(pr_logits.float(), dim=-1)
            # child score = parent slot's conditional log-prob of child token
            child_score = [0.0] * nb
            for i in range(1, nb):
                child_score[i] = float(logp[parent_big[i], nodes_big[i]])

            kept, remap = _reexpand(parent_big, child_score, keep_nodes)
            tokens = [nodes_big[o] for o in kept]
            depth = [depth_big[o] for o in kept]
            parent = [(-1 if parent_big[o] == -1 else remap[parent_big[o]]) for o in kept]
        else:
            tokens, depth, parent = nodes_big, depth_big, parent_big

        N = len(tokens)
        tree_sizes.append(N)
        tok_t = torch.tensor([tokens], device=target.device)
        depth_t = torch.tensor(depth, device=target.device)
        pos_t = (start + depth_t).unsqueeze(0)
        cache_pos = torch.arange(start, start + N, device=target.device)

        anc2 = torch.zeros((N, N), dtype=torch.bool)
        for i in range(N):
            j = i
            while j != -1:
                anc2[i, j] = True
                j = parent[j]
        mask = torch.full((1, 1, N, start + N), neg_inf, dtype=target.dtype, device=target.device)
        mask[..., :start] = 0
        mask[0, 0, :, start:] = torch.where(
            anc2.to(target.device), torch.tensor(0.0, device=target.device),
            torch.tensor(neg_inf, device=target.device),
        ).to(target.dtype)

        if draft_prefill and return_stats:
            draft_prefill = False
            decode_start = _cuda_time()

        # ---- 4. verify pruned tree in one target forward ----
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
        tgt_argmax = output.logits[0].argmax(dim=-1)

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
        m = len(accepted) - 1

        for k_i in range(1, len(accepted)):
            output_ids[0, start + k_i] = tokens[accepted[k_i]]
        output_ids[0, start + m + 1] = bonus
        committed = m + 1
        acceptance_lengths.append(committed)

        acc_t = torch.tensor(accepted, device=target.device)
        gather_idx = torch.cat([torch.arange(start, device=target.device), start + acc_t])
        for layer in past_key_values_target.layers:
            layer.keys = layer.keys[:, :, gather_idx, :].contiguous()
            layer.values = layer.values[:, :, gather_idx, :].contiguous()
        start += committed

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
        big_sizes=big_sizes,
        num_target_forwards=num_target_forwards,
        num_draft_forwards=num_draft_forwards,
    )
