# FiLM Text-Guided External Attention

> 基于论文 External Attention（Guo et al., TPAMI 2023）的改造版本，  
> 在 Multi-Head External Attention 的 memory 查询阶段引入 FiLM（Feature-wise Linear Modulation）机制，  
> 用文本特征动态调制视觉 attention map，实现 text-guided 的视觉自注意力。

---

## 动机

标准 External Attention 的 memory $M_k$ 是数据集级别的共享先验，对所有样本一视同仁。  
引入 FiLM 后，文本特征可以对 attention map 施加**仿射变换**（scale + shift），  
让不同语言描述动态地"激活"不同的视觉记忆槽，实现语言条件下的视觉特征聚合。

---

## 数学公式

### Step 0：输入投影 + 分头

给定视觉特征 $F \in \mathbb{R}^{B \times N \times C}$，文本全局特征 $t \in \mathbb{R}^{B \times C}$（对 token 序列取均值）：

$$F' = F W_{in} \in \mathbb{R}^{B \times N \times C'}, \quad C' = C \cdot \text{coef}$$

$$F_i = \text{reshape}(F') \in \mathbb{R}^{B \times H' \times N \times (C'/H')}, \quad H' = H \cdot \text{coef}$$

### Step 1：External Attention 查询（$M_k$ 投影）

$$\tilde{A} = F_i M_k^T \in \mathbb{R}^{B \times H' \times N \times S}$$

其中 $M_k \in \mathbb{R}^{S \times (C'/H')}$ 为共享外部 key memory，$S = C / \text{coef}$。

### Step 2：Padding Mask（可选）

$$\tilde{A}_{b,h,n,\cdot} = -\infty \quad \text{if token } n \text{ is padding}$$

### Step 3：FiLM 调制

文本特征经两个独立 MLP 生成仿射参数：

$$\gamma = \text{MLP}_\gamma(t) \in \mathbb{R}^{B \times S}, \quad \beta = \text{MLP}_\beta(t) \in \mathbb{R}^{B \times S}$$

对 attention logits 施加仿射变换：

$$\hat{A} = (1 + \gamma) \odot \tilde{A} + \beta$$

> 对比：无 FiLM 的 text-guided 版本使用 sigmoid gate：$\hat{A} = \sigma(\text{MLP}(t)) \odot \tilde{A}$，只能做抑制，不能做增强。  
> FiLM 的 $(1+\gamma)$ 允许 scale > 1，$\beta$ 允许 shift，表达能力更强。

### Step 4：Double Normalization

$$\bar{a}_{b,h,n,s} = \frac{\exp(\hat{A}_{b,h,n,s})}{\sum_{n'} \exp(\hat{A}_{b,h,n',s})} \quad \text{(col-wise softmax over } N \text{)}$$

$$a_{b,h,n,s} = \frac{\bar{a}_{b,h,n,s}}{\sum_{s'} \bar{a}_{b,h,n,s'}} \quad \text{(row-wise L1 norm over } S \text{)}$$

### Step 5：Memory 聚合 + 输出投影

$$\text{out}_i = A \cdot M_v \in \mathbb{R}^{B \times H' \times N \times (C'/H')}$$

$$F_{out} = \text{Concat}(\text{out}_1, \ldots, \text{out}_{H'}) \cdot W_o \in \mathbb{R}^{B \times N \times C}$$

---

## 伪代码

```python
# FiLM Text-Guided Multi-Head External Attention
#
# Input:
#   x          — [B, N, C]   visual features (batch_first=True)
#   text_feat  — [B, C]      global text feature (mean-pooled over tokens)
#   key_padding_mask — [B, N] True = padding position
#
# Learnable params:
#   W_in       — Linear(C, C*coef)
#   M_k        — Linear(C//H, S, bias=False)   shared key memory
#   M_v        — Linear(S, C//H, bias=False)   shared value memory
#   MLP_gamma  — Linear(C, C) -> ReLU -> Linear(C, S)
#   MLP_beta   — Linear(C, C) -> ReLU -> Linear(C, S)
#   W_o        — Linear(C*coef, C)
#
# Hyperparams:
#   H     — num_heads (original)
#   coef  — expansion factor
#   H'    = H * coef   (effective heads)
#   S     = C // coef  (memory size per head)

def FiLMTextGuidedExternalAttention(x, text_feat, key_padding_mask):
    B, N, C = x.shape

    # Step 0: project + split heads
    x = W_in(x)                              # [B, N, C*coef]
    x = x.view(B, N, H', C*coef//H')
    x = x.permute(0, 2, 1, 3)               # [B, H', N, C//H]

    # Step 1: external attention query
    attn = M_k(x)                            # [B, H', N, S]

    # Step 2: mask padding before softmax
    if key_padding_mask is not None:
        attn = attn.masked_fill(
            key_padding_mask[:, None, :, None], -inf
        )                                    # [B, H', N, S]

    # Step 3: FiLM modulation
    gamma = MLP_gamma(text_feat)             # [B, S]
    beta  = MLP_beta(text_feat)              # [B, S]
    gamma = gamma.view(B, 1, 1, S)
    beta  = beta.view(B, 1, 1, S)
    attn  = (1.0 + gamma) * attn + beta      # [B, H', N, S]

    # Step 4: double normalization
    attn = softmax(attn, dim=-2)             # col-wise over N
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)  # row L1 over S

    # Step 5: aggregate + output projection
    out = M_v(attn)                          # [B, H', N, C//H]
    out = out.permute(0, 2, 1, 3)           # [B, N, H', C//H]
    out = out.reshape(B, N, C * coef)        # [B, N, C*coef]
    out = W_o(out)                           # [B, N, C]
    return out
```

---

## 与原版 External Attention 的对比

| 特性 | 原版 MEA | Text-Guided (gate) | FiLM Text-Guided |
|------|---------|-------------------|-----------------|
| 文本条件 | 无 | sigmoid gate（只抑制） | affine $(1+\gamma, \beta)$（增强+抑制+偏移） |
| 参数量增加 | — | $+2 \times (C \to S)$ MLP | $+2 \times (C \to C \to S)$ MLP |
| 表达能力 | 数据集先验 | 语言条件抑制 | 语言条件仿射调制 |
| padding 处理 | 输入清零（旧） | 输入清零（旧） | softmax 前 $-\infty$ mask（修复后） |
