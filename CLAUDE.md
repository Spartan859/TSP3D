# TSP3D — External Attention 实验分支（TGEA）

3D Visual Grounding 项目，基于 CVPR'25 TSP3D（Text-guided Sparse Voxel Pruning）。当前工作分支 `TGEA`（Text-Guided External Attention），在原 TSP3D 基础上把 `BiEncoderLayer` 的 self-attention 替换为 External Attention 的各种变体，实验文本引导机制。

## 当前默认分支与实验焦点

- 默认分支：**`TGEA`**（不是 main）
- 实验目标：对比三类 attention 变体在 bi-encoder 各层的效果
  1. 原 self-attention（baseline）
  2. Plain External Attention（`--use_external_attn_bi_layer`）
  3. Text-guided EA with sigmoid gate（`--use_text_guided_external_attn_bi_layer`）
  4. FiLM Text-guided EA（`--use_film_text_guided_external_attn_bi_layer`，`(1+γ)·attn+β`）
- 最新训练命令：`scripts/experiments/EA_fixed/EA0.sh`（默认 4 卡，batch 28，linear LR scaling）
- 测试：在训练命令后加 `--checkpoint_path PATH --eval`

## 核心代码地图

| 模块 | 位置 | 作用 |
|------|------|------|
| 主训练入口 | `train_dist_mod.py` | `torchrun` 入口，内含 `TrainTester` |
| CLI & 配置 | `main_utils.py` | `parse_option()` 所有 CLI 参数都在这 |
| BeaUTyDETR 模型 | `models/bdetr.py` | 顶层模型 |
| **主 grounding head** | `models/multilevel_head_refine.py` | `TSPHead`，含 `_prune_inference`（:791）、`forward_test`（:1552）、3 个 `BiEncoderLayer` + `com_trans` |
| **Attention 模块** | `models/trans_modules.py` | `BiEncoderLayer`（:212）、`ExternalMultiheadAttention`（:304）、`CrossAttentionLayer`（:28）|
| 数据集 | `src/joint_det_dataset.py` | ScanRefer / NR3D / SR3D / WildRefer 多数据集 |
| 评估器 | `src/grounding_evaluator.py` | Top-k + IoU 阈值 |

## External Attention 关键参数

通过 CLI（`main_utils.py:177-197`）控制在哪几层 bi-encoder 启用哪种变体，值是**层索引列表**（0/1/2）：

```bash
--use_external_attn_bi_layer 0 1           # 层 0、1 用 plain EA
--use_text_guided_external_attn_bi_layer 0 # 层 0 用 gate-based text guided
--use_film_text_guided_external_attn_bi_layer 0  # 层 0 用 FiLM
--external_attn_coef 4                     # EA 内部的 head/dim 扩展系数
```

优先级：FiLM > text-guided > plain EA > self-attention。同一层多选时以最高级为准（在 `BiEncoderLayer.__init__` 的条件分支里）。

## 已知坑 / 踩过的雷

1. **`ExternalMultiheadAttention` 里 `self.k` 曾硬编码为 `256 // coef`**（与 `embed_dim` 无关）。已修正为 `embed_dim // coef`（`trans_modules.py:315`）。**副作用**：`embed_dim ∈ {128, 64}` 的旧 checkpoint（来自 `keep_trans`/`com_trans`，见 `multilevel_head_refine.py:425,433`）里 `linear_0/linear_1/text_mlp*` 形状与修正后不匹配，`load_state_dict` 会报 size mismatch。需要重训或者为兼容旧权重显式传 `k` 参数。

2. **Padding mask 全 True 触发 `softmax(-inf)=nan`**：batch size > 1 + 某样本该层无有效点时会发生（`multilevel_head_refine.py:1696` 的 `sampled_coords[:,:,0]==-1`）。修复在 `trans_modules.py:369` 的 `nan_to_num(0.0)`。

3. **FiLM 顺序**：`(1+γ)·attn + β` 必须在 `masked_fill(-inf)` **之前**，否则 `γ=-1` 时 `0*(-inf)=nan`。当前顺序正确（`trans_modules.py:354-367`）。

4. **Pruning 链路**：`ExternalMultiheadAttention → keep_conv → sigmoid → 1-score > threshold` 剪点。任何上游 nan 都会让所有点被剪掉并触发 `forward_test` 中的 `break`（`multilevel_head_refine.py:1586`）。日志关键词 `Prune inference removed all points`。

## 训练 / 测试

```bash
# 训练（自动选空闲 GPU，默认 4 卡）
bash scripts/experiments/EA_fixed/EA0.sh

# 指定 GPU
CVD=0,1,2,3 bash scripts/experiments/EA_fixed/EA0.sh

# 单卡调试
bash scripts/experiments/EA_fixed/EA0.sh -s

# 小数据集
bash scripts/experiments/EA_fixed/EA0.sh -m

# 评估（在命令行后追加）
bash scripts/experiments/EA_fixed/EA0.sh --checkpoint_path /path/to/ckpt.pth --eval
```

其他脚本：`scripts/train_scanrefer_single*.sh`、`scripts/test_*.sh`，数据集分别是 ScanRefer / NR3D / SR3D / WildRefer。

## 目录速查

- `scripts/experiments/` — 按实验分组的训练脚本；`EA_fixed/` 是当前主力，`inference_speed/` 是速度测试
- `EVAL_PARAMS.md` — 免重训就能调的评估期参数（`--com_threshold`、`--prune_threshold_0/1`、`--nms_pre` 等），含扫描建议和命令模板
- `experiments.md` — 推理速度对比表 + 日志路径
- `TSP3D_method_explanation.md` — TSP3D 原方法说明
- `multihead_external_attention.md/.drawio` — 原版 MEA 公式 + 图
- `film_text_guided_external_attn.md/.drawio` — FiLM 变体公式 + 图
- `use_film_text_guided_external_attn_bi_layer0.md/.drawio` — 在 layer 0 启用 FiLM 的架构图
- `Beyond_Self-Attention_*.pdf` — EA 原论文
- `data/` — ScanRefer/WildRefer/scannetv2 数据 + RoBERTa 权重
- `output/`, `log/`, `tensorboard_output/` — 训练产物
- `ckpt_scanrefer.pth` — 预训练 checkpoint
- `scripts/sync_to_zhangyu.sh` — 同步代码到协作者环境

## 协作规范（这个对话里学到的）

- 回答要简洁，不要结尾总结；数学公式用 **inline `$...$`** 而不是 `$$...$$`（后者会单开一行影响排版）
- drawio 文件要满足：`<mxGraphModel math="1" ...>` 才能渲染 LaTeX；`mxCell id` 必须是纯数字字符串，否则 draw.io 报 `d.setId is not a function`；drawio 里用 `\(...\)` 做 inline LaTeX
- 文件改动前先 `Read`，bug fix 要给出根本原因不要堆 try/except 掩盖
- 实验脚本里的 lr 已经按 `CUR_BS/BASE_BS` 线性缩放，改 bs 时不用手动调 lr
