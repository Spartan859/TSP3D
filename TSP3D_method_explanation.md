# TSP3D 工作区完整方法说明（3D Visual Grounding 与 Segmentation）

本文档基于当前工作区代码实现，对该方法做端到端拆解。重点覆盖：数据与监督构建、模型前向、跨模态交互、稀疏体素剪枝与补全、分割与细化分支、训练损失、推理与评估。

## 1. 方法定位

TSP3D 是一个面向 3D 视觉指代定位（Visual Grounding）的单阶段稀疏体素框架，并可选扩展到实例分割与框细化。

核心目标是：
- 在保持或提升 Grounding 精度的同时，提高推理效率。
- 将语言信息显式注入到 3D 稀疏体素处理链路中，用于“保留哪些体素、补全哪些体素”。

仓库 README 中对应论文是 Text-guided Sparse Voxel Pruning for Efficient 3D Visual Grounding。

## 2. 代码总览与模块分工

核心调用链如下：

- 训练入口与流程控制
  - train_dist_mod.py
  - main_utils.py
- 数据集与文本标注对齐
  - src/joint_det_dataset.py
  - get_gt.py
- 主模型装配
  - models/bdetr.py
- 3D 稀疏视觉主干
  - models/mink_resnet.py
- 多层 Head（含剪枝、补全、分割、Refine）
  - models/multilevel_head_refine.py
  - models/multilevel_head.py（不含 refine 的基线版本）
- 跨模态 Transformer/外部注意力
  - models/trans_modules.py
- 评估
  - src/grounding_evaluator.py

## 3. 训练与数据流主线

### 3.1 训练主入口

TrainTester 在 train_dist_mod.py 中定义，完成：
- 构建 Joint3DDataset
- 构建 BeaUTyDETR（实际是 TSP3D 风格模型）
- DDP 训练与验证循环
- 评估 Grounding 指标以及可选 mask 指标

训练时每个 batch 的核心调用是：
1. 从 batch_data 通过 get_gt.py 转换出 3D box、mask、辅助框等监督
2. 调用 model(inputs, gt...) 执行前向并返回损失字典
3. 反向传播 + 优化器更新

### 3.2 数据集与监督构建

Joint3DDataset 支持 ScanRefer、SR3D、NR3D 及 joint_det 混合训练策略。

每个样本返回内容包括：
- 点云 point_clouds
- 文本 utterances
- 主目标框与场景全体框
- 点级目标掩码 gt_masks
- 文本 token 对齐矩阵（positive_map、modify_positive_map、rel_positive_map 等）

文本对齐通过字符区间映射到 tokenizer token 区间，实现短语级监督信号。

get_gt.py 将批数据打包为 mmdet3d 可用的 DepthInstance3DBoxes，并生成：
- gt_bbox_new：主监督框
- gt_all_bbox_new：场景候选框
- auxi_bbox：辅助对象框（可为空）
- gt_masks_new：点级二值监督（按目标对象聚合）

## 4. 模型总体结构

## 4.1 顶层网络 BeaUTyDETR（models/bdetr.py）

整体由三部分组成：
- 视觉编码：Minkowski 稀疏 3D 主干 TSPBackbone
- 文本编码：冻结的 RoBERTa + 线性投影到 d_model=128
- 任务头：TSPHead 或 TSPHead_refine

输入是点云和文本，输出在训练时为损失字典，在测试时为检测框（及可选分割掩码）。

## 4.2 视觉编码（models/mink_resnet.py）

TSPBackbone 是基于 MinkowskiEngine 的 3D 稀疏 ResNet，典型过程：
- 稀疏卷积 conv1
- 可选池化
- 多 stage 残差块下采样
- 输出多尺度 sparse tensor 列表

这些多尺度特征会被 Head 的上采样路径逐级融合和筛选。

## 4.3 文本编码（models/bdetr.py）

文本处理流程：
1. RobertaTokenizerFast 做 batch tokenize
2. RobertaModel 提取 token hidden states
3. text_projector 将维度映射到 128
4. text_attention_mask 指示 padding token

文本编码器默认冻结，以降低训练不稳定性与显存成本。

## 5. TSP Head 机制（核心）

TSPHead_refine 在 models/multilevel_head_refine.py 中实现了方法主体。

### 5.1 多层稀疏上采样 + 语言引导筛选

Head 从深层稀疏特征向高分辨率逐层上采样，并在关键层执行：
- keep_trans：视觉-语言双向交互
- keep_conv：预测每个体素的保留分数
- pruning：按分数选择 Top-K 体素，抑制无关区域

训练时使用 _prune_training：
- 每个 batch 内每个场景取 topk，k 来自 pts_prune_threshold
- 可随机扰动阈值增强鲁棒性

测试时使用 _prune_inference：
- 按 sigmoid 分数阈值保留体素
- 若全部被删除会触发日志告警

### 5.2 completion 补全分支（com_trans）

在较高分辨率层，引入 com_trans 与 com_cls：
- 对候选体素与文本做交互，预测补全得分
- 从原始浅层特征中采样出高置信新增体素
- 与当前特征合并，形成“剪枝 + 补全”的平衡

直观理解：
- keep 分支负责删除噪声
- com 分支负责找回被剪掉但语言相关的区域

### 5.3 全局文本融合到检测特征

MinkowskiFeatureFusionBlock 将文本全局特征（通常是首 token）复制到每个体素，并与体素特征拼接后做 1x1 稀疏卷积。

随后输出：
- bbox_conv 回归框参数
- cls_conv 预测分类/置信

## 6. 分割分支与外部自注意力

当 use_seg 开启时，Head 会从检测路径特征再上采样得到分割特征：
- upsample_st_2 与 upsample_st_4 恢复空间分辨率
- conv_32_ch 统一到 32 通道
- seg_unet 生成点级分割 logit

当 use_seg_external_self_attn 开启时，会在分割上采样前插入外部自注意力：
- 先做位置编码 PositionEmbeddingLearned
- 用 PosTransformerEncoderLayerNoFFN 做自注意力
- 支持 text-guided external attention 与 FiLM 方式调制

这部分实现位于：
- TSPHead._seg_external_self_attn
- trans_modules.py 中的 ExternalMultiheadAttention

## 7. Refine 分支（ROI 网格池化 + 二阶段轻量细化）

当 use_refine 开启时，流程为：
1. 从 seg 前特征提取 ROI 内体素（RoIAwarePool3d）
2. 形成固定网格特征 (R, g, g, g, C)
3. 送入 SimpleRefineHead：
   - 稀疏卷积 trunk
   - 全连接 head
   - 输出 bbox_delta 与 score_logit

### 7.1 编解码方式

给定 ROI 框 r=(c_r, s_r) 与 GT 框 g=(c_g, s_g)，回归目标为：

- 中心偏移
  delta_c = (c_g - c_r) / s_r
- 尺寸偏移
  delta_s = log(s_g / s_r)

推理解码：
- c_hat = c_r + delta_c * s_r
- s_hat = s_r * exp(delta_s)

### 7.2 Refine 监督

- 框回归损失：SmoothL1(delta_pred, delta_target)
- 质量分支：BCEWithLogits，目标来自 IoU 引导质量映射

质量映射函数 iou_guided_quality：
- iou > 0.75 时目标为 1
- iou < 0.25 时目标为 0
- 中间线性映射 q = 2*iou - 0.5

## 8. 损失函数总览

在 TSPHead_refine._loss 中，最终损失由多项组成：

- bbox_loss：3D 框回归损失（AxisAlignedIoULoss2 / DIoU）
- cls_loss：分类 focal loss
- keep_loss：体素保留分支损失
- com_loss：补全分支分类损失
- seg_loss：分割 focal loss（可选）
- seg_loss_dice：分割 Dice loss（可选）
- refine_bbox_loss：Refine 框回归（可选）
- refine_score_loss：Refine 质量分数（可选）

代码里通过不同权重组合这些项。

## 9. 推理流程

forward_test 主要步骤：
1. 多层上采样过程中执行 keep 推理剪枝
2. 在高分辨率层执行 com 补全并合并体素
3. 输出检测框并做后处理
4. 若 use_seg，生成实例 mask
5. 若 use_refine，用 refine_delta 对 ROI 框做最终修正

最终输出：
- 3D box 结果（中心、尺寸、分数、标签）
- 可选实例分割掩码

## 10. 评估协议

GroundingEvaluator（src/grounding_evaluator.py）统计：
- IoU@0.25 与 IoU@0.5
- Top-k（当前常用 Top-1）
- easy/hard、unique/multi、view-dependent 等细分子集
- 开启分割时额外统计 mask 指标

## 11. 关键训练开关与超参

来自 main_utils.py 常用选项：
- use_seg：是否启用分割分支
- use_refine：是否启用 Refine 分支
- use_seg_external_self_attn：分割前是否插外部注意力
- use_external_attn_bi_layer：在哪些 BiEncoder 层启用 external attention
- voxel_size：稀疏体素尺寸
- mink_conv1_stride：主干第一层步长

此外，优化器将参数分组，给 keep_trans、text_encoder、seg/refine 分支设置独立学习率。

## 12. 与传统两阶段 3DVG 的差异（实现视角）

从代码实现上看，该方法区别于常见两阶段范式的关键点：
- 主路径是稀疏体素单阶段检测，不依赖重型 proposal-then-rank
- 语言在中间层直接参与体素筛选（keep）与补全（com）
- 分割与 refine 是可插拔增强分支，不破坏主检测路径
- 通过稀疏算子与逐层剪枝显著压缩推理计算量

## 13. 一句话总结

该工作区实现的是一个“语言驱动的稀疏体素动态路由器”：
- 在多尺度稀疏 3D 特征中，用文本决定哪些体素应被保留、哪些应被补全，
- 再通过可选分割与轻量 Refine 对结果精修，
- 最终兼顾 Grounding 精度与推理效率。

## 14. 建议阅读顺序（代码）

1. train_dist_mod.py
2. models/bdetr.py
3. models/multilevel_head_refine.py
4. models/trans_modules.py
5. src/joint_det_dataset.py
6. get_gt.py
7. src/grounding_evaluator.py
