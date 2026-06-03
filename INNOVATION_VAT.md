# Amazing Idea：Verification-Aware Training —— 把 draft 直接「为树验证器」训练

> 一句话：我们唯一的正收益（创新1 树draft，+1.13x）被 **draft 的 top-k coverage**
> 卡住，而 DFlash/EAGLE 都只用 per-token CE 训练 **top-1**。既然现在能训练了，就
> **闭环**——用 target 的树验证结果当奖励，**on-policy 地训练 draft 去最大化"被接受的
> 路径长度"**，而不是最大化单 token 准确率。直击瓶颈、且没人在 diffusion draft 上做过。

---

## 1. 为什么是这个（基于我们自己的实证）

| 已知事实（PROTOTYPE_RESULTS.md） | 推论 |
|---|---|
| 单序列 target verify 在 ≤64 token **成本平坦** | 多候选树验证几乎免费 → 创新1 成立 |
| 树 accepted/fwd：6.34→8.10→(128节点)8.61 后 tok/s 不再涨 | 收益上限 = **top-k coverage**，不是验证成本 |
| 创新2/B 提升 per-position 质量 = 负结果 | **接受长度由 prefix 前沿决定**，块尾 CE 无用 |
| DFlash/EAGLE 训练 = per-token CE | 只优化 **top-1**，与「树取 top-k 让 target 选」**目标不一致** |

**训练目标与部署目标错配**：我们部署的是树验证器（要的是「gold 落在某条 top-k 路径里」），
但训练的是「每个位置 top-1 对」。把这两者对齐，就能在**不增加任何推理成本**的前提下，
把树的 accepted/fwd 上限往上推，并让**更小的树**就达到现在大树的接受长度（更省）。

---

## 2. 核心方法：On-Policy Verification-Aware Training（VAT）

闭环训练，三件套我们都已经有（`train.py` / `tree.py` / `gen_data.py`）：

```
 当前 draft ──build tree──▶ 树候选(top-k×depth)
        ▲                         │
        │                    ONE target verify forward（和推理同款，便宜）
        │                         │
        │             每个节点：target 贪心是否接受 + 真正的 target 续写 token
        └──── 用「接受奖励 / 真续写」反传，训练 draft 的 top-k 分布 ────┘
```

**损失（两项，可单独消融）：**

1. **Coverage / 期望接受长度损失（主）**
   对每个前沿深度 d，设 target 在该前沿真正会接受的 token 为 `y_d*`（来自同一 target
   verify forward，**无额外成本**）。最大化它落入 draft top-k 的概率。可微代理：
   ```
   L_cov = Σ_d  w_d · CE(draft_logits_d, y_d*) · 1[y_d* 未在 draft top-k]   (hard-negative 加权)
         + λ_margin · Σ_d max(0, m − (s(y_d*) − s(k-th competitor)))        (margin: 把 gold 挤进 top-k)
   ```
   直觉：CE 已经把 top-1 训好；这里**专门修「gold 掉到 top-k 之外」的样本**——正是树
   coverage 的漏点。

2. **排序校准损失（次）**
   树是按**联合边际概率**堆排序建的（`tree.py::_build_tree`）。如果 draft 的概率排序与
   「真实被接受概率」不一致，节点预算就浪费在没用的分支上。加一个 pairwise ranking
   损失，让「更可能被接受的候选」边际概率更高：
   ```
   L_rank = Σ_{a accepted, b rejected}  softplus( s(b) − s(a) )
   ```

**总损失**：`L = L_ce(loss-decay) + α·L_cov + β·L_rank`，`α,β` 退火。
当 `α=β=0` 时**精确退化为我们现在的 CE 训练**（安全基线，可 A/B）。

**无损性**：训练目标变了，但**推理仍是 target 校验**——提交 token 永远来自 target，
所以**天然无损**。我们只是让 draft 更聪明地"投喂"树，验证器不变。

---

## 3. 为什么 amazing（novelty + 杠杆）

- **训练/推理协同设计**：EAGLE-2/3 用静态/动态树但**仍是 token-CE 训练**；DFlash 同理。
  「**为你部署的那个树验证器训练 draft**」在 block-diffusion draft 上**没有先例**。
- **直击唯一正收益的天花板**：我们已经知道树 work、且被 coverage 卡住——这是把
  8.10 → 逼近/突破 8.61 上限，并让 **64 节点达到 128 节点效果**（更快）。
- **几乎零新增推理成本**：奖励来自**和推理同款的一次 target forward**；部署形态不变。
- **可证伪、可量化**：直接用 `bench_tree.py` 量 accepted/fwd 与 tok/s，对照参考 draft。

---

## 4. 落地计划（全部复用现有代码，增量很小）

| 步骤 | 复用 | 新增 |
|---|---|---|
| 1. on-policy 数据/奖励采集 | `tree.py` 的树构建 + 4D mask + 一次 target verify | 把「每节点 accept/reject + target 续写」落盘成训练信号 |
| 2. VAT 损失 | `train.py::compute_loss` | 加 `L_cov`、`L_rank`（`--vat-weight`/`--rank-weight`），`within`/top-k 已有 |
| 3. 训练 | `scripts/train_full.sh` | 加 `--vat-*` 开关；先 warm-start 自现有 draft 再 VAT 微调 |
| 4. 评测 | `bench_tree.py` | acc/fwd、tok/s、固定树预算下 coverage 曲线；无损对照 top_k=1 |

**消融**（讲清楚是哪一项 work）：
- A) 纯 CE（基线，现状） B) +L_cov C) +L_cov+L_rank D) on-policy vs teacher-forced 数据。
- 主指标：**固定 64 节点预算下的 accepted/fwd 与 tok/s**；以及「达到 ref 128-节点
  accept 所需的最小节点数」（省多少）。

**里程碑式的成功标准**：64 节点树，accepted/fwd 从 8.10 提到 ~8.6+（即用一半预算拿到
现在大树的接受长度），tok/s 从 1.13x 提到 ~1.2x+，且 top_k=1 仍 token-identical（无损）。

---

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| coverage 代理不可微（top-k 硬判定） | 用 hard-negative 加权 CE + margin 软化；不直接对 top-k 求导 |
| on-policy 采集慢（每步一次 verify） | 与推理同成本；可离线批量预生成树+奖励缓存（像 gen_data 一样分片多卡）|
| 过度 coverage 伤 top-1（接受率掉） | `α` 退火 + 保留 loss-decay CE 主导；A/B 守住基线 |
| 分布漂移（teacher-forced→on-policy） | DAgger 式混采：早期 teacher-forced，逐步增大 on-policy 比例 |

---

## 6. 备选方向（次优，列出备查）

- **B. 学习型自适应树预算**：用 draft 置信度预测「这一步该铺多宽的树」，把固定预算变成
  按需分配（难块多铺、易块少铺）。与 VAT 正交，可叠加。
- **C. block=32 + early-stop 训练**：我们有 pipeline，但 headroom 分析显示只有当 p90
  接受触顶时才显著——增量，不 amazing。
- **D. 树内精炼（救活创新2）**：refine 的收益只有在多候选验证下才能兑现，把创新2 放进
  树的深层候选里。属于 VAT 的子模块。

---

## 7. 建议

做 **VAT（§2）**。它把我们**唯一的正收益**和**刚建好的训练能力**接在一起，机制清晰、
无损、推理零额外成本、用现有 `bench_tree.py` 即可量化，且在 diffusion draft 上是新的。
先做最小闭环：teacher-forced 数据 + `L_cov` 单项消融，量 64-节点 accepted/fwd 是否上抬；
若正向，再加 on-policy 与 `L_rank`。
