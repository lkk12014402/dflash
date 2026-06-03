# DFlash 训练代码详解（`dflash/train.py`）

本文讲清楚我们实现的训练代码：**做什么、原理是什么、调用链怎么走、每段代码细节**。
配套脚本与启动命令见 `scripts/README.md`；本文聚焦实现本身。

---

## 0. 一句话概括

DFlash 的 draft 是一个 **feature-conditioned block-diffusion**（特征条件化的块扩散）模型：
它**不重新处理 token**，而是 **cross-attend 到冻结 target 抽取的 hidden 特征**，在一次
**双向（非因果）forward** 里，把一个「全 MASK 的块」一次性去噪成接下来的
`block_size − 1` 个 token。

训练的唯一目标：**让训练 forward 与推理 forward 在数值上严格等价**——“训的就是推的”。
`selftest` 子命令是这条不变量的正确性闸门（100% argmax 一致才算过）。

---

## 1. 先理解推理（训练要对齐的对象）

来自 `dflash/model.py` 的 `dflash_generate`。对一个起点在绝对位置 `s` 的块：

- **context** = target 特征 `F_0 … F_{s-1}`（严格在 s 之前的所有位置）。
- draft 看到的 **noise 块** = `[x_s, MASK, MASK, …, MASK]`，长度 `B = block_size`：
  - 块内位置 0（`x_s`）是已被接受的「干净 token」（推理里的 bonus token）；
  - 位置 `1..B-1` 是 MASK，待预测。
- 每个 draft 层内部：`q = noise(B)`；`k = v = concat([ctx_features, noise])`；
  注意力 **非因果**——每个 noise token 看到「所有 context + 同块所有 noise」（双向）。
- 在 MASK 位置 `s+i`（i=1..B-1）用 `target.lm_head` 预测 token `x_{s+i}`
  （logits 取自 `draft_out[:, 1-B:, :]`）。

关键实现细节（`Qwen3DFlashAttention.forward`，model.py:211）：
`k = cat([k_proj(target_hidden), k_proj(noise)])`，所以 key 长度是 `ctx_len + q_len`；
RoPE 里（`apply_rotary_pos_emb`，model.py:176）**query 只取 `cos[..., -q_len:, :]`**，
key 用完整 `cos`。这两点决定了训练侧 `position_ids` 必须怎么构造（见 §4）。

---

## 2. 训练目标与原理

teacher-force 一条**位于 target 自身解码轨迹上**的序列 `x`（用 `gen_data.py` 生成
target 贪心续写）。这一点至关重要：当数据就是 target 贪心输出时，**对 gold token 做硬
交叉熵 == 蒸馏温度 0 的贪心 verifier**，因此无需额外 KL 也能对齐 target。

整条训练 forward 分 4 步：

1. 跑一次冻结 target，取每个位置的特征 `F_i`（与推理同一个
   `extract_context_feature`）。
2. 把 completion 区域按 `block_size` 切块（带**随机相位 offset** 提升对齐覆盖），
   把每块的尾部全部置 MASK。
3. 跑**一次整序列 draft forward**，用一个 4D 注意力 mask 精确复现「逐块推理」的可见性：
   ```
   query i（块起点 s）：ctx 列 j 可见 ⟺ j < s
                        noise 列 j 可见 ⟺ block(j) == block(i)
   ```
4. 只在 MASK 位置算交叉熵（可选 + 一个**正确移位**的 KL 蒸馏）。

只训练 draft 参数；target 的 `embed_tokens` 与 `lm_head`（tied）保持冻结并共享。

**为什么「整序列一次 forward」等价于「逐块多次 forward」**：4D mask 让块之间互不可见，
每个块的 query 只能看到自己块的 noise + 严格在前的 context 特征；这与逐块推理时
「context=F_{0..s-1}、noise=本块」给出的可见集合**完全一致**。唯一的数值差异来自
sdpa/lm_head 的归约顺序（key 数量不同走不同 kernel），属于浮点噪声，不改变 argmax。

---

## 3. 块与 mask 构造（等价性的核心）

### `build_block_ids(seq_len, prompt_len, block_size, offset)` — train.py:80
给每个位置一个「块起点 id」。`[0, prompt_len+offset)` 为纯 context（哨兵 `_PREFIX=-1`，
不计损失）；其余按 `block_size` 平铺，某位置的 id = 它所在块第一个 token 的绝对下标。
```python
block_ids[start:] = start + ((pos - start) // block_size) * block_size
```

### `build_attention_mask(block_ids, dtype, device)` — train.py:97
产出 **4D 加性 mask `[1,1,L,2L]`**，对应 key 排布 `[ctx(L) ; noise(L)]`：
- ctx 半区：`j < block_start(i)`（对干净前缀特征是因果的）；
- noise 半区：`block_ids[j] == block_ids[i]` 且是真实块（`>=0`，排除 prefix/pad）。

可见处填 0，不可见填 `finfo(dtype).min`（加到 logits 上等价于 −∞）。

### `build_example(input_ids, prompt_len, block_size, mask_token_id, offset)` — train.py:118
- `offset` 不给则 `random.randint(0, block_size-1)`（每个样本随机相位，每个 epoch 经
  DataLoader 重洗实现重采样，覆盖各种对齐）。
- 块内位置 0 是**干净真实 token**（推理的 bonus token），位置 1..B-1 替换成
  `mask_token_id`、作为预测目标。
- 返回 `ids, noise_ids, labels, block_ids`；`labels` 除 MASK 位外全为 −100。

哨兵：`_PREFIX=-1`（纯 context，不监督）、`_PAD=-2`（右 padding，永不可见、永不监督）。

---

## 4. 核心 forward

### `TokenizedDataset` / `Collator` — train.py:141 / 168
- 数据是 JSONL：`{"input_ids":[...], "prompt_len":P}`。过滤掉
  `completion < min_completion(=2)` 的退化样本。
- `Collator` 右 padding 到 batch 最长；构造 `target_attn`（target 的 padding mask），
  padding 位 `block_ids=_PAD`、`labels=-100`。

### `compute_target_features(target, input_ids, target_attn, layer_ids, want_logits)` — train.py:200
`@torch.no_grad` 跑冻结 target，`output_hidden_states=True`，再用
`extract_context_feature(out.hidden_states, layer_ids)` 取 5 个目标层特征拼成 `[B,L,5H]`。
（`extract_context_feature` 用 offset+1，把 `target_layer_ids` 映射到 37 元 hidden_states
元组里的 `[2,10,18,26,34]`，与推理一致。）

### `dflash_training_forward(draft, target, batch)` — train.py:216
这是等价性关键，逐行：
```python
feats, _   = compute_target_features(...)          # [B,L,5H] -> draft.dtype
noise_emb  = target.get_input_embeddings()(noise_ids)   # 复用 target 的(冻结)词嵌入
masks      = stack([build_attention_mask(block_ids[b]) ...])   # [B,1,L,2L]
position_ids = cat([arange(L), arange(L)]).expand(B,-1)        # 长度 2L！
hidden = draft(target_hidden=feats, noise_embedding=noise_emb,
               position_ids=position_ids, attention_mask=masks,
               use_cache=False, is_causal=False)               # [B,L,H]
# 只在被监督位置过 lm_head（省显存）
loss_mask    = labels.view(-1) != -100
logits       = target.lm_head(hidden.reshape(-1,H)[loss_mask]) # [N,V]
within       = (abs_pos - block_ids)[loss_mask]               # 块内位置 k ∈ 1..B-1
```
要点：
- **`position_ids` 长度是 2L**：因为 key=`[ctx;noise]`，而 RoPE 对 query 只取后半段
  `cos[-q_len:]`（即 noise 段）。`arange(L)` 复制两遍，正好让 ctx 段用前缀位置、noise
  段用各自的绝对位置，和推理逐块时 `pos = arange(0, s+bk)` 一致。
- **`noise_embedding` 用 target 的词嵌入**：draft 与 target 共享 tied embedding/lm_head。
- **先 gather 再 lm_head**：词表 15 万，只对被监督的 N 个位置投影，显存友好。
- **`within`**（块内第 k 个位置）：供 loss-decay 加权与 per-position 准确率统计。

### `draft(...)` 内部（model.py）
`DFlashDraftModel.forward` → 每层 `Qwen3DFlashAttention.forward`：
`q=q_proj(noise)`；`k=cat([k_proj(ctx), k_proj(noise)])`、`v` 同理；
`apply_rotary_pos_emb` 对 q 取 `cos[-q_len:]`、对 k 取全量；最后 sdpa + 我们传入的 4D mask。

---

## 5. 损失：`compute_loss` — train.py:258

```python
logits, labels, loss_mask, hidden, within = dflash_training_forward(...)
# (1) loss-decay 加权 CE（DFlash eq.4）
w  = exp(-(within-1)/gamma)            # k 越大权重越小；gamma 默认 16
ce = (w * CE_tok).sum() / w.sum()
# (2) 可选 KL 蒸馏（默认关）
#     draft 在绝对位置 i 预测 x_i，其 verifier 分布来自 TARGET 的 i-1 位置
shifted = flat_pos[loss_mask] - 1
kl = KL(log_softmax(logits/T), softmax(t_logits[shifted]/T)) * T^2
loss = (1-distill_weight)*ce + distill_weight*kl
# (3) 统计：token_acc 及 per-position acc@{1,2,4,8}
```
原理与坑点：
- **loss-decay**：块内早错会让后面全废，所以越靠前的位置权重越大（`w_k=exp(-(k-1)/γ)`）。
- **KL 的 off-by-one**：draft 位置 i 的目标是 `x_i`，对应 target 的「预测 `x_i`」分布在
  **位置 i-1** 的 logits。错位会系统性地训歪，这里用 `flat_pos-1` 对齐。
- **CE 已等价温度0蒸馏**：数据是 target 贪心，故默认 `distill_weight=0` 就够；KL 是可选增强。
- **`acc@1` 是 accept rate 的最紧代理**：紧邻 anchor 的第一个 MASK 位最容易对、最影响接受长度；
  全局 `token_acc` 把块尾难位平均进来，会显著偏低、具有误导性。

---

## 6. 等价性自检：`selftest` — train.py:312

构造一条随机序列（prompt P=7，offset=3，2 个块），分别跑：
- **(B)** 整序列训练式 forward（`dflash_training_forward`）；
- **(A)** 逐块推理式 forward：对每个块用 `target_hidden=feats[:, :s]`、noise=本块、
  `position_ids=arange(0, s+bk)`，logits 取 `hid_A[:, 1-bk:, :]`。

比较两者在每个被预测位置的 logits：
- **硬闸门**：argmax 一致率 > 99.9%；
- 容差：fp32 `max|Δlogits| < 3e-2`（bf16 放宽到 0.2）。差异仅来自 sdpa/lm_head 归约顺序
  （A、B 的 key 数量不同 → 不同 kernel），非逻辑错误。

实测：**fp32 100% argmax 一致，max|Δ|≈0.0096**。这就是「训练==推理」的证明。

运行：
```bash
PYTHONPATH=$PWD python -m dflash.train selftest \
  --model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16 --fp32
```

---

## 7. 训练循环：`train` — train.py:419

1. **分布式**：`RANK/WORLD_SIZE` 在环境里就 `init_process_group("nccl")`；
   `torch.cuda.set_device(local_rank)`；种子 `seed+rank`。
2. **模型**：target bf16 冻结（`requires_grad_(False)`、`.eval()`）；draft 二选一——
   `--init-from`（从已有 draft 继续）或 `--draft-config`（从配置 **fresh** 初始化）。
   只有 draft 进 `train()`。
3. **数据**：`TokenizedDataset` + `DistributedSampler`（DDP 时）+ `Collator`，`drop_last=True`。
4. **优化**：`AdamW(betas=(0.9,0.95))`；step 数 = `ceil(len(loader)/grad_accum) * epochs`；
   **warmup + cosine** 衰减（`lr_lambda`）。
5. **DDP**：`nn.parallel.DistributedDataParallel`；`core` 始终指向未包裹模型，用于
   `optimizer`、`clip_grad_norm_`、`save_pretrained`。
6. **wandb**（仅 rank0、可选）：`wb.init(config=vars(args))`，每 `log_every` 记录
   `train/{loss,ce,kl,token_acc,acc@1,acc@2,acc@4,acc@8,lr,tok_per_s}`，结束 `wb.finish()`。
7. **循环**：`autocast(bf16)` 下 `compute_loss` → `(loss/grad_accum).backward()` →
   每 `grad_accum` 步做 `clip_grad_norm_` + `optim.step()` + `sched.step()`。
8. **保存**：可选 `--save-every` 周期 ckpt；每 epoch 存 `epoch{e}`；最后存 `final`
   （`save_pretrained`，draft 可直接被 benchmark 加载）。

---

## 8. 完整调用链（一次训练 step）

```
train(args)                                   # train.py:419
└─ for batch in loader:                        # Collator 输出 dict
   └─ compute_loss(draft, target, batch, …)    # train.py:258
      ├─ dflash_training_forward(draft, target, batch)         # :216
      │  ├─ compute_target_features(target, …)                 # :200 冻结 target forward
      │  │  └─ extract_context_feature(hidden_states, ids)     # model.py:39
      │  ├─ target.get_input_embeddings()(noise_ids)           # 共享词嵌入
      │  ├─ build_attention_mask(block_ids[b], …)  ×B          # :97 4D mask
      │  ├─ draft(target_hidden, noise_embedding, position_ids,# model.py forward
      │  │        attention_mask, is_causal=False)
      │  │  └─ Qwen3DFlashAttention.forward                    # model.py:211
      │  │     ├─ k=v=cat([proj(ctx), proj(noise)])
      │  │     └─ apply_rotary_pos_emb(q取cos[-L:], k取全量)   # model.py:176
      │  └─ target.lm_head(hidden[loss_mask])                  # 只投影被监督位
      ├─ loss-decay 加权 CE  (+ 可选 shifted-KL)
      └─ 统计 token_acc / acc@{1,2,4,8}
   └─ backward → (每 grad_accum) clip → optim.step → sched.step → wandb.log
```

---

## 9. 设计取舍速查

| 决策 | 原因 |
|------|------|
| 整序列单 forward + 4D mask | 等价逐块推理但**一次算完**，训练高效；selftest 保证数值一致 |
| 数据用 target 贪心续写 | 让硬 CE == 蒸馏温度0 verifier，无需 KL 即对齐 target（无损前提）|
| `position_ids` 长 2L、复制两遍 | 适配 model.py 的 RoPE（q 取后半 / k 取全量）与 `k=[ctx;noise]` 排布 |
| 先 gather 再 lm_head | 15 万词表，只投影被监督位，省大量显存 |
| loss-decay `exp(-(k-1)/γ)` | 块内早错废全块，加权强调靠前位置（论文 eq.4）|
| KL 用 target 的 i-1 logits | 修正 off-by-one，预测 `x_i` 的分布来自 target 位置 i-1 |
| 共享 target 的 embed/lm_head 且冻结 | tied 权重；draft 只学中间的去噪 transformer |
| `acc@1` 作为主指标 | 最接近 accept rate；全局 token_acc 被块尾难位拉低、易误判 |

---

> 参考：`dflash/model.py`（推理实现，请勿修改）、DFlash 论文 §4.2（KV 注入、随机 anchor +
> masked 块尾、跨块不可见稀疏 mask、冻结共享 embed/lm_head、损失衰减加权）、EAGLE-2/3
> （teacher-forcing 训练范式与数据来源）。
