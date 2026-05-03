# 免重训练可调评估参数

加载已有 checkpoint 后，以下参数可在推理时调整，无需重训。

---

## 参数总览

| 参数 | CLI flag | 默认值 | 代码位置 | 是否已暴露 CLI |
|------|----------|--------|----------|--------------|
| CBA 补全阈值 | `--com_threshold` | 0.15 | `multilevel_head_refine.py:1657` | ✅ |
| CBA 采样点数 | `--num_samples_com` | 2400 | `multilevel_head_refine.py:1622` | ✅ |
| 剪枝阈值 layer0 | `--prune_threshold_0` | 0.3 | `multilevel_head_refine.py:817` | ✅ |
| 剪枝阈值 layer1 | `--prune_threshold_1` | 0.7 | `multilevel_head_refine.py:817` | ✅ |
| NMS 候选框数 | `--nms_pre` | 1 | `multilevel_head_refine.py:1537` | ✅ |
| NMS IoU 阈值 | `--nms_iou_thr` | 0.5 | `multilevel_head_refine.py:1498` | ✅ |
| NMS score 阈值 | `--nms_score_thr` | 0.01 | `multilevel_head_refine.py:1484` | ✅ |
| 随机种子 | `--rng_seed` | 42 | `multilevel_head_refine.py:1623` | ✅ |

---

## 详细说明

### 1. `--com_threshold`（CBA 补全阈值）

**作用**：CBA（Completion Branch Attention）对 finest-level 点云做补全预测，`sigmoid(pred) > threshold` 的点才被加回稀疏张量。

- 更低 → 补更多点 → 更密集的 finest-level 特征 → 可能提升 recall
- 更高 → 补更少点 → 更快但可能漏掉关键点

**建议扫描**：`0.05, 0.10, 0.15, 0.20, 0.30, 0.50`

**预期提升**：0.1~0.5 pp（对 CBA 依赖强的场景影响更大）

**注意**：该参数在训练时也用于 `forward_train`，但 checkpoint 权重不依赖它的具体值，可以自由调。

---

### 2. `--num_samples_com`（CBA 采样点数）

**作用**：每个场景从 finest-level 随机采样多少个候选点送入 `com_trans`（CBA 的 BiEncoderLayer）。超出部分随机截断，不足部分 zero-padding。

- 更多 → 覆盖更多候选点 → 补全更准 → 显存和时间线性增加
- 更少 → 更快但可能漏掉稀疏区域的点

**建议扫描**：`1200, 2400, 4800, 9600`

**预期提升**：0.1~0.3 pp（场景点云密度越不均匀，提升越明显）

**注意**：`com_trans` 的 attention 复杂度是 $O(N \cdot k)$，$N$ = `num_samples_com`，增大 4x 时间约增 4x（该模块）。

---

### 3. `--prune_threshold_0` / `--prune_threshold_1`（剪枝阈值）

**作用**：UNet 解码器在 layer 0 和 layer 1 做稀疏剪枝，`1 - sigmoid(keep_score) > threshold` 的点被保留（注意是反向：score 越低越被保留）。

- 更低 → 保留更多点 → 更密集的中间特征 → 可能提升精度但显存增加
- 更高 → 更激进剪枝 → 更快但可能丢失关键点

**建议扫描**：
- `prune_threshold_0`：`0.2, 0.3, 0.4, 0.5`（layer 0，默认 0.3）
- `prune_threshold_1`：`0.5, 0.6, 0.7, 0.8`（layer 1，默认 0.7）

**预期提升**：0.2~0.8 pp（最有潜力的参数，直接影响 grounding 用到的点云密度）

**注意**：如果阈值过低导致所有点被剪掉，会触发 `Prune inference removed all points` 警告并中断推理。

---

### 4. `--nms_pre`（NMS 候选框数）

**作用**：在 NMS 前按 max score 保留 top-k 个候选框。默认 1 意味着直接取最高分框，不做 NMS。

- 设为 >1（如 5, 10, 20）→ 允许多个候选框进入 NMS → 可能找到更好的框
- 设为 1 → 当前行为，直接取最高分

**建议扫描**：`1, 5, 10, 20`（配合 `--nms_iou_thr` 一起调）

**预期提升**：0~0.3 pp（取决于模型是否产生多个高质量候选框）

**注意**：当前模型在 grounding 任务中通常只需要 1 个框，`nms_pre > 1` 的收益不确定。

---

### 5. `--nms_iou_thr` / `--nms_score_thr`

**作用**：NMS 的 IoU 阈值和 score 过滤阈值，仅在 `nms_pre > 1` 时生效。

**建议**：先把 `nms_pre` 调到 >1 再扫这两个。

---

### 6. `--rng_seed`（随机种子 / 伪 TTA）

**作用**：`num_samples_com` 采样用 `torch.randperm`，不同 seed 采不同点子集。

**伪 TTA 方案**：对同一 checkpoint 用多个 seed 跑 eval，取每个样本得分最高的预测框（需要修改 evaluator 或后处理脚本）。

**建议**：`0, 1, 2, 3, 4`，5 次推理后 voting。

**预期提升**：0.1~0.3 pp（低方差，但几乎零代码改动）

---

## 推荐扫描顺序

```
1. 确认基线（默认参数 eval）
2. 扫 com_threshold（5 次）→ 找最优值
3. 在最优 com_threshold 下扫 num_samples_com（4 次）
4. 扫 prune_threshold_0 × prune_threshold_1（4×4=16 次，可先粗扫）
5. 如有余力：nms_pre 扫描 + 多 seed TTA
```

---

## 评估命令模板

```bash
# 用 20260216 实验的 epoch-204 checkpoint 评估
bash scripts/experiments/20260216/text_guided_EA_FiLM_totalB28.sh \
    --checkpoint_path scripts/experiments/20260216/scanrefer/2026-02-16_11-14-34/ckpt_epoch_204.pth \
    --eval \
    --com_threshold 0.15 \
    --num_samples_com 2400 \
    --prune_threshold_0 0.3 \
    --prune_threshold_1 0.7

# 用 EA_fixed/EA0.sh 系列 checkpoint 评估
bash scripts/experiments/EA_fixed/EA0.sh \
    --checkpoint_path /path/to/ckpt.pth \
    --eval \
    --com_threshold 0.10 \
    --num_samples_com 4800 \
    --prune_threshold_0 0.25 \
    --prune_threshold_1 0.65
```

---

## 注意事项

1. **旧 checkpoint 兼容性**：`20260216` 系列 checkpoint 用旧 CLI（`--use_*_bi_layer0` 布尔 flag），eval 时需要确保脚本里有这三个 flag。`EA_fixed` 系列用新 CLI（`--use_*_bi_layer 0 2`）。

2. **`prune_threshold` 与 `com_threshold` 的交互**：剪枝后剩余点越多，CBA 补全的增量越小（因为已经很密了）。两者联合调优时注意这个 trade-off。

3. **显存**：`num_samples_com` 增大和 `prune_threshold` 降低都会增加中间张量大小，注意 OOM。
