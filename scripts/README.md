# DFlash 训练流水线 — 脚本与启动命令

本目录 `scripts/` 提供 DFlash draft 模型的**数据生成 → 合并 → 训练**全流程脚本。
所有脚本都从仓库根目录运行，自动设置 `PYTHONPATH`，参数通过**环境变量**覆盖。

> 关键不变量：draft 始终无损（提交 token 来自 target 校验）；训练 forward 与推理
> forward 严格一致（`dflash.train selftest` 是正确性闸门）。

---

## 0. 正确性自检（先跑这个）

验证「训练 forward == 推理 forward」（100% argmax 一致即通过）：

```bash
cd /data/lkk/mtp/dflash
PYTHONPATH=$PWD python -m dflash.train selftest \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --fp32
```

---

## 1. 脚本总览

| 脚本 | 作用 |
|------|------|
| `scripts/gen_full.sh`   | 多卡分片生成 target-greedy 训练数据 |
| `scripts/merge_data.sh` | 合并分片、清洗、统计 |
| `scripts/train_full.sh` | 4 卡 DDP 全量训练（默认 fresh-from-config，wandb 日志）|
| `scripts/run_6k.sh`     | 一键复现 6000 样本的小规模验证run（gen+merge+train）|

数据来源 `--source`：`codealpaca`（20K）、`ultrachat`（开放）、`nemotron-v2`（gated，
按 chat/code/math/stem 类别拉取，需 HF 访问权限）。官方配方 ≈ 800K：Nemotron-v2 +
CodeAlpaca，response 由 target 重新生成（target-aligned）。

---

## 2. 全量训练（真·800K，约 3–4 天）

> 实测吞吐 ≈ 0.3s/样本（4 卡并行，bs=24, max_new_tokens=512）。800K 生成 ≈ 66h，
> 之后训练又是数十小时。脚本用 `nohup` 后台常驻，断连不中断。

### 2.1 生成（GPUs 4-7）

```bash
cd /data/lkk/mtp/dflash
GPUS="4,5,6,7" \
SOURCE="codealpaca,ultrachat,nemotron-v2" \
MAX_SAMPLES=800000 PER_SOURCE=400000 \
MAX_NEW_TOKENS=512 BATCH_SIZE=24 \
OUT_DIR=cache/full \
bash scripts/gen_full.sh
```

监控：`tail -f logs/gen_full/gen0.log`
输出：`cache/full/train_part{0..3}.jsonl`

### 2.2 合并

```bash
OUT_DIR=cache/full MERGED=cache/full/train_all.jsonl \
bash scripts/merge_data.sh
```

### 2.3 训练（4 卡 DDP，fresh-from-config，wandb）

```bash
DATA=cache/full/train_all.jsonl \
GPUS="4,5,6,7" \
OUTPUT_DIR=dflash_ckpt_full \
EPOCHS=3 BATCH_SIZE=4 GRAD_ACCUM=4 LR=1e-4 LOSS_DECAY=16 \
SAVE_EVERY=2000 \
WANDB=1 WANDB_PROJECT=dflash-draft WANDB_RUN_NAME=full-b16 \
bash scripts/train_full.sh
```

监控：`tail -f logs/full-b16.log` 或 wandb 面板。
ckpt：周期性 `dflash_ckpt_full/step{N}`、每 epoch `epoch{e}`、最终 `final`。

---

## 3. 小规模验证run（6000 样本，一键）

用于证明训练管线正确（loss 8.2→~5.6，acc@1 上升），**质量远低于发布模型**（数据少两个数量级）。

```bash
cd /data/lkk/mtp/dflash
bash scripts/run_6k.sh
# 复用已有数据、只重训：
SKIP_GEN=1 bash scripts/run_6k.sh
```

等价的手动三步：

```bash
# gen
GPUS="4,5,6,7" SOURCE="codealpaca,ultrachat" \
  PER_SOURCE=3000 MAX_SAMPLES=6000 MAX_NEW_TOKENS=384 BATCH_SIZE=24 \
  OUT_DIR=cache/run6k LOG_DIR=logs/run6k bash scripts/gen_full.sh
# merge
OUT_DIR=cache/run6k MERGED=cache/run6k/train_all.jsonl bash scripts/merge_data.sh
# train (fresh, 4 epochs, wandb)
DATA=cache/run6k/train_all.jsonl OUTPUT_DIR=dflash_ckpt_scratch \
  EPOCHS=4 BATCH_SIZE=4 GRAD_ACCUM=4 LR=1e-4 LOSS_DECAY=16 LOG_EVERY=10 \
  WANDB_RUN_NAME=scratch-b16-6k MASTER_PORT=29541 bash scripts/train_full.sh
```

---

## 4. 关键参数说明

**生成 (`gen_full.sh`)**
- `MAX_SAMPLES` 全局样本上限（所有分片合计）；每片取 `shard_id :: num_shards`。
- `PER_SOURCE` 每个数据源最多拉取的 prompt 数。
- `SEED` 各分片必须一致（保证全局池 shuffle 相同，分片不重叠不漏）。
- `MAX_NEW_TOKENS` / `BATCH_SIZE` 直接影响吞吐与显存。

**训练 (`train_full.sh`)**
- 默认 **fresh-from-config**（`DRAFT_CONFIG=z-lab/Qwen3-4B-DFlash-b16`）；
  设 `INIT_FROM=<draft_ckpt>` 改为从已有 draft 继续。
- `LOSS_DECAY`（默认 16）：DFlash 损失衰减权重 `w_k=exp(-(k-1)/γ)`，强调 block 内靠前位置。
- `DISTILL_WEIGHT`>0 开启对 target logits 的 KL 蒸馏（默认 0，即对 target 贪心 token 做 CE）。
- `SAVE_EVERY` 长跑periodic ckpt（步数），0 关闭。
- `WANDB=0` 关闭 wandb。

**wandb 指标**：`train/loss`、`train/ce`、`train/kl`、`train/token_acc`、
`train/acc@{1,2,4,8}`（block 内第 k 个 mask 位的准确率，**acc@1 是 accept rate 的最紧代理**）、
`train/lr`、`train/tok_per_s`。

---

## 5. 训练后评测

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD python -m dflash.benchmark \
  --backend transformers \
  --model Qwen/Qwen3-4B \
  --draft-model dflash_ckpt_full/final
```

诚实对照：小规模run不会达到发布模型的 τ≈6.5；全量 800K 才接近官方质量。
