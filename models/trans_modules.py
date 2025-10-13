import pdb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from Swin3D.modules.swin3d_layers import BasicLayer
import MinkowskiEngine as ME
import torch

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

# BRIEF Cross-attention between language and vision
class CrossAttentionLayer(nn.Module):
    """Cross-attention between language and vision."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8,
                 dim_feedforward=256, use_butd_enc_attn=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.use_butd_enc_attn = use_butd_enc_attn

        # Cross attention from lang to vision
        self.cross_lv = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_lv = nn.Dropout(dropout)
        self.norm_lv = nn.LayerNorm(d_model)
        self.ffn_lv = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm_lv2 = nn.LayerNorm(d_model)

        # Cross attention from vision to lang
        self.cross_vl = deepcopy(self.cross_lv)
        self.dropout_vl = nn.Dropout(dropout)
        self.norm_vl = nn.LayerNorm(d_model)
        self.ffn_vl = deepcopy(self.ffn_lv)
        self.norm_vl2 = nn.LayerNorm(d_model)

        if use_butd_enc_attn:
            self.cross_d = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_d = nn.Dropout(dropout)
            self.norm_d = nn.LayerNorm(d_model)

    def forward(self, vis_feats, vis_key_padding_mask, text_feats,
                text_key_padding_mask, pos_feats,
                detected_feats=None, detected_mask=None):
        """Forward pass, vis/pos_feats (B, V, F), lang_feats (B, L, F)."""
        # produce key, query, value for image
        qv = kv = vv = vis_feats
        qv = qv + pos_feats  # add pos. feats only on 【query】

        # produce key, query, value for text
        qt = kt = vt = text_feats

        # step cross attend language to vision
        text_feats2 = self.cross_lv(
            query=qt.transpose(0, 1),
            key=kv.transpose(0, 1),
            value=vv.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=vis_key_padding_mask  # (B, V)
        )[0].transpose(0, 1)
        text_feats = text_feats + self.dropout_lv(text_feats2)
        text_feats = self.norm_lv(text_feats)
        text_feats = self.norm_lv2(text_feats + self.ffn_lv(text_feats))

        # step cross attend vision to language
        vis_feats2 = self.cross_vl(
            query=qv.transpose(0, 1),
            key=kt.transpose(0, 1),
            value=vt.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=text_key_padding_mask  # (B, L)
        )[0].transpose(0, 1)
        vis_feats = vis_feats + self.dropout_vl(vis_feats2)
        vis_feats = self.norm_vl(vis_feats)

        # step cross attend vision to boxes
        if detected_feats is not None and self.use_butd_enc_attn:
            vis_feats2 = self.cross_d(
                query=vis_feats.transpose(0, 1),
                key=detected_feats.transpose(0, 1),
                value=detected_feats.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=detected_mask
            )[0].transpose(0, 1)
            vis_feats = vis_feats + self.dropout_d(vis_feats2)
            vis_feats = self.norm_d(vis_feats)

        # FFN
        vis_feats = self.norm_vl2(vis_feats + self.ffn_vl(vis_feats))

        return vis_feats, text_feats
    
class TransformerEncoderLayerNoFFN(nn.Module):
    """TransformerEncoderLayer but without FFN."""

    def __init__(self, d_model, nhead, dropout, use_external_attn=False):
        """Intialize same as Transformer (without FFN params)."""
        super().__init__()
        if use_external_attn:
            self.self_attn = ExternalMultiheadAttention(d_model, nhead, attn_drop=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer (same as parent class).

        Args:
            src: (S, B, F)
            src_mask: the mask for the src sequence (optional)
            src_key_padding_mask: (B, S) mask for src keys per batch (optional)
        Shape:
            see the docs in Transformer class.
        Return_shape: (S, B, F)
        """
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

# BRIEF vision self-attention
class PosTransformerEncoderLayerNoFFN(TransformerEncoderLayerNoFFN):
    """TransformerEncoderLayerNoFFN but additionaly add pos_embed in query."""

    def __init__(self, d_model, nhead, dropout, use_external_attn=False):
        """Intialize same as parent class."""
        super().__init__(d_model, nhead, dropout, use_external_attn)

    def forward(self, src, pos, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer (same as parent class).

        Args:
            src: (S, B, F)  
            pos: (S, B, F) positional embeddings
            src_mask: the mask for the src sequence (optional)
            src_key_padding_mask: (B, S) mask for src keys per batch (optional)
        Shape:
            see the docs in Transformer class.
        Return_shape: (S, B, F)
        """
        src2 = self.self_attn(
            src + pos, src + pos, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src
    
# BRIEF vision text self attention and cross attention
class BiEncoderLayer(nn.Module):
    """Self->cross layer for both modalities."""

    def __init__(self, d_model=256, dropout=0.1, activation="relu", n_heads=8,
                 dim_feedforward=256,
                 self_attend_lang=True, self_attend_vis=True,
                 use_butd_enc_attn=False,
                 use_external_attn=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        # self attention in language
        if self_attend_lang:
            self.self_attention_lang = TransformerEncoderLayerNoFFN(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout,
                use_external_attn=False
            )
        else:
            self.self_attention_lang = None

        # self attention in vision
        if self_attend_vis:
            self.self_attention_visual = PosTransformerEncoderLayerNoFFN(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout,
                use_external_attn=use_external_attn
            )
        else:
            self.self_attention_visual = None

        # cross attention in language and vision
        self.cross_layer = CrossAttentionLayer(
            d_model, dropout, n_heads, dim_feedforward,
            use_butd_enc_attn
        )
    
    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points={}, detected_feats=None,
                detected_mask=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        # STEP 1. Self attention for vision
        if self.self_attention_visual is not None:
            vis_feats = self.self_attention_visual(
                vis_feats.transpose(0, 1),
                pos_feats.transpose(0, 1),
                src_key_padding_mask=padding_mask
            ).transpose(0, 1)

        # STEP 2. Self attention for language
        if self.self_attention_lang is not None:
            text_feats = self.self_attention_lang(
                text_feats.transpose(0, 1),
                src_key_padding_mask=text_padding_mask
            ).transpose(0, 1)

        # STEP 3. Cross attention
        vis_feats, text_feats = self.cross_layer(
            vis_feats=vis_feats,
            vis_key_padding_mask=padding_mask,
            text_feats=text_feats,
            text_key_padding_mask=text_padding_mask,
            pos_feats=pos_feats,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )

        return vis_feats, text_feats


# BRIEF 
class BiEncoder(nn.Module):
    """Encode jointly language and vision."""

    def __init__(self, bi_layer, num_layers):
        """Pass initialized BiEncoderLayer and number of such layers."""
        super().__init__()
        self.layers = _get_clones(bi_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points={},
                detected_feats=None, detected_mask=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        for i, layer in enumerate(self.layers):
            vis_feats, text_feats = layer(
                vis_feats,
                pos_feats,
                padding_mask,
                text_feats,
                text_padding_mask,
                end_points,
                detected_feats=detected_feats,
                detected_mask=detected_mask
            )
            if 'lv_attention' in end_points:
                end_points['lv_attention%d' % i] = end_points['lv_attention']
        return vis_feats, text_feats
    
# BRIEF vision text self attention and cross attention
class BiEncoderLayerSwin(nn.Module):
    """Self->cross layer for both modalities."""

    def __init__(self, d_model=256, dropout=0.1, activation="relu", n_heads=8,
                 dim_feedforward=256,
                 self_attend_lang=True, self_attend_vis=True,
                 use_butd_enc_attn=False, window_size=9, quant_size=4, 
                 swin_layer_num=2,
                 use_F3_CA=False,
                 swin_drop_path=0.0):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        self.use_F3_CA = use_F3_CA
        
        # self attention in language
        if self_attend_lang:
            self.self_attention_lang = TransformerEncoderLayerNoFFN(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout
            )
        else:
            self.self_attention_lang = None

        # self attention in vision
        if self_attend_vis:
            # self.self_attention_visual = PosTransformerEncoderLayerNoFFN(
            #     d_model=d_model,
            #     nhead=n_heads,
            #     dropout=dropout
            # )
            # pdb.set_trace()
            self.self_attention_visual = BasicLayer(
                dim=d_model,
                depth=swin_layer_num,
                num_heads=n_heads,
                window_size=window_size,
                quant_size=quant_size,
                drop_path=swin_drop_path,
                cRSE="XYZ"
            )
            self.visual_norm = nn.LayerNorm(d_model)
            
        else:
            self.self_attention_visual = None
        # cross F3
        self.cross_layer_F3 = CrossAttentionLayer(
            d_model, dropout, n_heads, dim_feedforward,
            use_butd_enc_attn
        )
        # cross attention in language and vision
        self.cross_layer = CrossAttentionLayer(
            d_model, dropout, n_heads, dim_feedforward,
            use_butd_enc_attn
        )

    def forward(self, vis_feats, pos_feats, coords_vis, sampled_coords, x_F3_vis_feats, x_F3_pos_feats, padding_mask, x_F3_padding_mask, text_feats,
                text_padding_mask, end_points={}, detected_feats=None,
                detected_mask=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        
        # STEP 1. Self attention for vision
        if self.self_attention_visual is not None:
            # 先用vis_feats和sampled_coords构建SparseTensor(去除[-1,-1,-1,-1])
            # vis_feats: [B, N, C], sampled_coords: [B, N, 4]
            B, N, C = vis_feats.shape
            device = vis_feats.device
            # 展平
            vis_feats_flat = vis_feats.view(-1, C)
            sampled_coords_flat = sampled_coords.view(-1, 4)
            # 有效mask
            valid_mask = (sampled_coords_flat != -1).all(dim=1)
            feats_valid = vis_feats_flat[valid_mask]
            coords_valid = sampled_coords_flat[valid_mask]
            
            # 去除重复坐标，只保留第一个
            coords_valid_np = coords_valid.cpu().numpy()
            _, unique_idx_np = np.unique(coords_valid_np, axis=0, return_index=True)
            unique_idx = torch.from_numpy(unique_idx_np).to(coords_valid.device)

            # Use the indices to get the unique tensors
            coords_valid = coords_valid[unique_idx]
            feats_valid = feats_valid[unique_idx]
            # 同步修改valid_mask（只保留unique的True）
            valid_mask_indices = valid_mask.nonzero(as_tuple=True)[0]
            keep_idx = valid_mask_indices[unique_idx]
            new_valid_mask = torch.zeros_like(valid_mask)
            new_valid_mask[keep_idx] = True
            valid_mask = new_valid_mask
            sp_tensor = ME.SparseTensor(
                features=feats_valid,
                coordinates=coords_valid,
                device=device
            )
            # coords_vis: [B, N, C']
            # 构建 coords_sp，和 sp_tensor 对齐
            coords_vis_flat = coords_vis.view(-1, coords_vis.shape[-1])
            coords_sp = ME.SparseTensor(
                features=coords_vis_flat[valid_mask],
                coordinates=coords_valid,
                device=device
            )
            _, new_sp, _ = self.self_attention_visual(sp_tensor,coords_sp)
            # new_sp还原到vis_feats
            # new_sp.C: [M, 4]，new_sp.F: [M, C]
            # coords_valid: [num_valid, 4]，valid_mask: [B*N]
            # 目标：还原到vis_feats_flat的顺序（无效位置补0），再reshape成[B, N, C]
            
            vis_feats_flat_out = torch.zeros_like(vis_feats_flat)
            # 构建映射：coords_valid 在 vis_feats_flat 的索引就是 valid_mask.nonzero(as_tuple=True)[0]
            idx_valid = valid_mask.nonzero(as_tuple=True)[0]
            if not (sp_tensor.F.shape[0] == new_sp.F.shape[0] and idx_valid.shape[0] == new_sp.F.shape[0]):
                print(f"sp_tensor.F.shape: {sp_tensor.F.shape}")
                print(f"new_sp.F.shape: {new_sp.F.shape}")
                print(f"idx_valid.shape: {idx_valid.shape}")
                pdb.set_trace()
            vis_feats_flat_out[idx_valid] = new_sp.F
            vis_feats = vis_feats_flat_out.view(B, N, C)
            # pdb.set_trace()
            vis_feats = self.visual_norm(vis_feats)
            # pdb.set_trace()

        # STEP 2. Self attention for language
        if self.self_attention_lang is not None:
            text_feats = self.self_attention_lang(
                text_feats.transpose(0, 1),
                src_key_padding_mask=text_padding_mask
            ).transpose(0, 1)
            # pdb.set_trace()

        # 新增：用x_F3_vis_feats和x_F3_pos_feats对vis_feats再做一次cross attention
        # x_F3_vis_feats: [B, N, C], x_F3_pos_feats: [B, N, C]
        if self.use_F3_CA:
            vis_feats, x_F3_vis_feats = self.cross_layer_F3(
                vis_feats=vis_feats,
                vis_key_padding_mask=padding_mask,
                text_feats=x_F3_vis_feats+x_F3_pos_feats,
                text_key_padding_mask=x_F3_padding_mask,
                pos_feats=pos_feats,
                detected_feats=None,
                detected_mask=None
            )
            # pdb.set_trace()
        
        # STEP 3. Cross attention
        vis_feats, text_feats = self.cross_layer(
            vis_feats=vis_feats,
            vis_key_padding_mask=padding_mask,
            text_feats=text_feats,
            text_key_padding_mask=text_padding_mask,
            pos_feats=pos_feats,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )

        

        return vis_feats, text_feats, x_F3_vis_feats
    
class BiEncoderSwin(nn.Module):
    """Encode jointly language and vision."""

    def __init__(self, bi_layer, num_layers):
        """Pass initialized BiEncoderLayer and number of such layers."""
        super().__init__()
        self.layers = _get_clones(bi_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, vis_feats, pos_feats, coords_vis, sampled_coords, x_F3_vis_feats, x_F3_pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points={},
                detected_feats=None, detected_mask=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        for i, layer in enumerate(self.layers):
            vis_feats, text_feats = layer(
                vis_feats,
                pos_feats,
                coords_vis,
                sampled_coords,
                x_F3_vis_feats,
                x_F3_pos_feats,
                padding_mask,
                text_feats,
                text_padding_mask,
                end_points,
                detected_feats=detected_feats,
                detected_mask=detected_mask
            )
            if 'lv_attention' in end_points:
                end_points['lv_attention%d' % i] = end_points['lv_attention']
        return vis_feats, text_feats
    
class ExternalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_drop=0., proj_drop=0., batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.coef = 4
        self.trans_dims = nn.Linear(embed_dim, embed_dim * self.coef)
        self.num_heads_eff = self.num_heads * self.coef
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(embed_dim * self.coef // self.num_heads_eff, self.k)
        self.linear_1 = nn.Linear(self.k, embed_dim * self.coef // self.num_heads_eff)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim * self.coef, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.batch_first = batch_first

    def forward(self, query, key=None, value=None, attn_mask=None, key_padding_mask=None, need_weights=False):
        # 只支持自注意力（query=key=value），忽略attn_mask
        # 输入: 
        # batch_first=False: (N, B, C)
        # batch_first=True: (B, N, C)
        # key_padding_mask: (B, N)  True表示padding
        x = query
        if not self.batch_first:
            x = x.transpose(0, 1)  # (B, N, C)
        if key_padding_mask is not None:
            if key_padding_mask.shape[0] != x.shape[0] or key_padding_mask.shape[1] != x.shape[1]:
                raise ValueError(f"key_padding_mask shape {key_padding_mask.shape} does not match input shape {(x.shape[0], x.shape[1])}")
            mask = (~key_padding_mask).unsqueeze(-1).float()  # (B, N, 1)
            x = x * mask
        B, N, C = x.shape
        x = self.trans_dims(x) # B, N, C'
        x = x.view(B, N, self.num_heads_eff, -1).permute(0, 2, 1, 3)
        attn = self.linear_0(x)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        attn_output_weights = None
        if not self.batch_first:
            x = x.transpose(0, 1)  # (N, B, C)
        return x, attn_output_weights