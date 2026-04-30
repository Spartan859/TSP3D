# Multi-Head External Attention (MEA)

> 论文：Beyond Self-Attention: External Attention Using Two Linear Layers for Visual Tasks  
> Guo et al., IEEE TPAMI 2023

---

## 数学公式

### 基础 External Attention（单头）

给定输入特征 $F \in \mathbb{R}^{N \times d}$，外部可学习记忆 $M_k, M_v \in \mathbb{R}^{S \times d}$（$S \ll N$）：

**Step 1：计算注意力图**

$$\tilde{A} = F M_k^T \in \mathbb{R}^{N \times S}$$

**Step 2：Double Normalization（先列方向 softmax，再行方向 L1 norm）**

$$\hat{a}_{ij} = \frac{\exp(\tilde{a}_{ij})}{\sum_k \exp(\tilde{a}_{kj})} \quad \text{(column-wise softmax, over } N \text{)}$$

$$a_{ij} = \frac{\hat{a}_{ij}}{\sum_k \hat{a}_{ik}} \quad \text{(row-wise L1 norm, over } S \text{)}$$

**Step 3：聚合输出**

$$F_{out} = A M_v \in \mathbb{R}^{N \times d}$$

---

### Multi-Head External Attention

将输入 $F$ 分成 $H$ 个头，每头维度为 $d/H$，所有头**共享**同一对记忆 $M_k, M_v \in \mathbb{R}^{S \times (d/H)}$：

$$h_i = \text{ExternalAttention}(F_i,\ M_k,\ M_v), \quad i = 1, \ldots, H$$

$$F_{out} = \text{MultiHead}(F,\ M_k,\ M_v) = \text{Concat}(h_1, \ldots, h_H)\, W_o$$

其中 $W_o \in \mathbb{R}^{d \times d_{in}}$ 为输出投影矩阵。

> **关键区别**：标准 multi-head self-attention 每个头有独立的 $K_i, V_i$；MEA 所有头共享 $M_k, M_v$，既减少参数量，又让所有头能通过共享记忆隐式交互。

---

## 伪代码

### Algorithm 1：Single-Head External Attention

```python
# Input:  F    — [B, N, C]
# Params: M_k  — Linear(C, S, bias=False)
#         M_v  — Linear(S, C, bias=False)
# Output: out  — [B, N, C]

def ExternalAttention(F, M_k, M_v):
    F = query_linear(F)          # [B, N, C]
    attn = M_k(F)                # [B, N, S]  (F @ M_k^T)
    attn = softmax(attn, dim=1)  # column-wise softmax (over N)
    attn = l1_norm(attn, dim=2)  # row-wise L1 norm (over S)
    out = M_v(attn)              # [B, N, C]
    return out
```

### Algorithm 2：Multi-Head External Attention

```python
# Input:  F    — [B, N, C_in]  (batch, pixels, channels)
# Params: M_k  — shared Linear(C//H, S, bias=False)
#         M_v  — shared Linear(S, C//H, bias=False)
#         H    — number of heads
#         W_o  — Linear(C, C_in)
# Output: out  — [B, N, C_in]

def MultiHeadExternalAttention(F, M_k, M_v, H, W_o):
    B, N, C_in = F.shape

    # 1. Input projection
    F = query_linear(F)              # [B, N, C]

    # 2. Split into H heads
    F = F.view(B, N, H, C // H)     # [B, N, H, C//H]
    F = F.permute(0, 2, 1, 3)       # [B, H, N, C//H]

    # 3. Compute raw attention via shared external key memory
    attn = M_k(F)                    # [B, H, N, S]  (F @ M_k^T)

    # 4. Double normalization
    attn = softmax(attn, dim=2)      # column-wise softmax (over N)
    attn = l1_norm(attn, dim=3)      # row-wise L1 norm (over S)

    # 5. Aggregate via shared external value memory
    out = M_v(attn)                  # [B, H, N, C//H]

    # 6. Merge heads
    out = out.permute(0, 2, 1, 3)   # [B, N, H, C//H]
    out = out.view(B, N, C)         # [B, N, C]

    # 7. Output projection
    out = W_o(out)                   # [B, N, C_in]
    return out
```

---

## 关键设计对比

| 特性 | Self-Attention | External Attention |
|------|---------------|-------------------|
| 复杂度 | $O(dN^2)$ | $O(dSN)$，$S \ll N$，线性复杂度 |
| K/V 来源 | 输入自身 | 共享外部记忆 $M_k, M_v$ |
| 多头参数 | 每头独立 $K_i, V_i$ | 所有头共享 $M_k, M_v$ |
| 跨样本关系 | 无（仅单样本内） | 隐式编码（记忆在全数据集上训练） |
| 归一化 | Softmax | Double Norm（列 softmax + 行 L1） |
| 典型 $S$ 值 | — | 64（实验中效果好） |

---

## 示意图

见同目录 `multihead_external_attention.drawio`，可用 [draw.io](https://app.diagrams.net) 或 VS Code Draw.io 插件打开。

流程概览：

```
Input F [B,N,C_in]
    ↓ query_linear
F [B,N,C]
    ↓ view + permute
F_i [B,H,N,C//H]
    ↓ @ M_k^T  ← Shared M_k [S, C//H]
Attn [B,H,N,S]
    ↓ softmax(dim=2) + l1_norm(dim=3)
A [B,H,N,S]
    ↓ @ M_v    ← Shared M_v [S, C//H]
out_i [B,H,N,C//H]
    ↓ permute + view
[B,N,C]
    ↓ W_o
Output F_out [B,N,C_in]
```
