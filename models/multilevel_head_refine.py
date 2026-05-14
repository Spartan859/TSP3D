import numpy as np
from typing import List

import MinkowskiEngine as ME

import torch,time
from mmcv.ops import nms3d, nms3d_normal
from torch import nn

from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
from .axis_aligned_iou_loss import AxisAlignedIoULoss2
from mmdet.models.losses import FocalLoss, DiceLoss
from .trans_modules import (BiEncoder, BiEncoderLayer, PositionEmbeddingLearned,
                            PosTransformerEncoderLayerNoFFN)
from .mink_unet import MinkUNet14B
from .roiaware_pool3d_utils import RoIAwarePool3d

import pdb
import logging

logger = logging.getLogger(__name__)

class MinkowskiFeatureFusionBlock(nn.Module):
    """
    Block to fuse backbone features with text features in Minkowski space.
    """
    def __init__(self, backbone_channels, text_channels, output_channels, dimension=3):
        super(MinkowskiFeatureFusionBlock, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            backbone_channels + text_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            dimension=dimension
        )
        self.norm = ME.MinkowskiBatchNorm(output_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, backbone_feats, text_feats):
        # Extract batch indices from the coordinates of backbone features
        batch_indices = backbone_feats.C[:, 0].long()  # Last column is batch index
        
        # Repeat text features for each point in the corresponding batch
        repeated_text_feats = text_feats[batch_indices]  # Use indexing to repeat text features
        
        # Combine the backbone and text features
        combined_features = torch.cat([backbone_feats.F, repeated_text_feats], dim=1)
        combined_feats = ME.SparseTensor(
            features=combined_features,
            coordinate_map_key=backbone_feats.coordinate_map_key,
            coordinate_manager=backbone_feats.coordinate_manager
        )
        
        # Convolution and normalization
        x = self.conv(combined_feats)
        x = self.norm(x)
        return self.relu(x)

class SimpleRefineHead(nn.Module):
    def __init__(self, in_ch=32, grid_size=6, hidden=256, mid_ch=64, D=3):
        super().__init__()
        self.grid_size = grid_size
        self.mid_ch = mid_ch
        self.D = D

        # -------- sparse conv trunk --------
        self.sparse_conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_ch, mid_ch, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(mid_ch),
            ME.MinkowskiReLU(inplace=True),
        )
        self.sparse_conv2 = nn.Sequential(
            ME.MinkowskiConvolution(mid_ch, mid_ch, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(mid_ch),
            ME.MinkowskiReLU(inplace=True),
        )
        self.sparse_pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)

        self.sparse_conv3 = nn.Sequential(
            ME.MinkowskiConvolution(mid_ch, mid_ch, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(mid_ch),
            ME.MinkowskiReLU(inplace=True),
        )
        self.sparse_conv4 = nn.Sequential(
            ME.MinkowskiConvolution(mid_ch, mid_ch, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(mid_ch),
            ME.MinkowskiReLU(inplace=True),
        )

        pooled_grid = grid_size // 2
        fc_in_dim = mid_ch * pooled_grid * pooled_grid * pooled_grid

        # -------- fc heads --------
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.bbox_fc = nn.Linear(hidden, 6)
        self.score_fc = nn.Linear(hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.bbox_fc.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_fc.bias, 0)

        nn.init.normal_(self.score_fc.weight, mean=0, std=0.001)
        nn.init.constant_(self.score_fc.bias, 0)

        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    @staticmethod
    def fake_sparse_idx(device, batch_size_rcnn):
        # 避免某些 batch/roi 全空导致 BN 或 sparse conv 出问题
        sparse_idx = torch.zeros((batch_size_rcnn, 4), dtype=torch.int32, device=device)
        sparse_idx[:, 0] = torch.arange(batch_size_rcnn, device=device, dtype=torch.int32)
        return sparse_idx

    def dense_to_sparse_with_fake(self, pooled_feats):
        """
        pooled_feats: (R, g, g, g, C)
        return:
            x_sp: ME.SparseTensor
            R: number of rois
        """
        R, gx, gy, gz, C = pooled_feats.shape
        device = pooled_feats.device

        # 非零体素
        valid_mask = pooled_feats.abs().sum(dim=-1) > 0   # (R, g, g, g)
        sparse_idx = valid_mask.nonzero(as_tuple=False)   # (N, 4): [roi_id, x, y, z]

        # ---- 关键：检查哪些 roi 一个点都没有，给它补一个 fake voxel ----
        if sparse_idx.numel() > 0:
            exist_roi = torch.unique(sparse_idx[:, 0])
        else:
            exist_roi = torch.empty((0,), dtype=torch.long, device=device)

        all_roi = torch.arange(R, device=device, dtype=torch.long)
        missing_roi_mask = torch.ones(R, dtype=torch.bool, device=device)
        if exist_roi.numel() > 0:
            missing_roi_mask[exist_roi] = False
        missing_roi = all_roi[missing_roi_mask]

        if missing_roi.numel() > 0:
            fake_idx = torch.zeros((missing_roi.numel(), 4), dtype=torch.long, device=device)
            fake_idx[:, 0] = missing_roi   # 放在每个缺失 roi 的 (0,0,0)
            sparse_idx = torch.cat([sparse_idx, fake_idx], dim=0)

        feats = pooled_feats[
            sparse_idx[:, 0].long(),
            sparse_idx[:, 1].long(),
            sparse_idx[:, 2].long(),
            sparse_idx[:, 3].long()
        ]  # (N, C)

        coords = sparse_idx.int().contiguous()

        x_sp = ME.SparseTensor(
            features=feats.contiguous(),
            coordinates=coords,
            tensor_stride=(1, 1, 1),
            device=device
        )
        return x_sp, R

    def sparse_to_dense(self, x_sp, batch_size_rcnn):
        """
        x_sp.dense() -> (B, C, X, Y, Z)
        """
        # pdb.set_trace()
        x_dense = x_sp.dense()[0]  # MinkowskiEngine 返回一般是 (tensor, min_coord, tensor_stride)
        # x_dense: (R, C, gx, gy, gz)
        return x_dense

    def forward(self, pooled_feats):
        """
        pooled_feats: (R, g, g, g, C)
        """
        R = pooled_feats.shape[0]
        if R == 0:
            device = pooled_feats.device
            return (
                torch.zeros((0, 6), device=device, dtype=pooled_feats.dtype),
                torch.zeros((0, 1), device=device, dtype=pooled_feats.dtype),
            )

        x_sp, R = self.dense_to_sparse_with_fake(pooled_feats)

        x_sp = self.sparse_conv1(x_sp)
        x_sp = self.sparse_conv2(x_sp)
        x_sp = self.sparse_pool(x_sp)
        x_sp = self.sparse_conv3(x_sp)
        x_sp = self.sparse_conv4(x_sp)

        pooled_grid = self.grid_size // 2
        dense_shape = torch.Size([R, self.mid_ch, pooled_grid, pooled_grid, pooled_grid])
        x_dense = x_sp.dense(shape=dense_shape)[0]
        # x_dense = x_sp.dense()[0]   # (B, C, 7, 7, 7)  或者视你的 ME 版本而定

        # 再做一层保险
        if x_dense.shape[0] != R:
            raise RuntimeError(
                f"RefineHead batch mismatch: expected R={R}, got dense batch={x_dense.shape[0]}"
            )

        x = x_dense.contiguous().view(R, -1)
        feat = self.shared_fc(x)
        bbox_delta = self.bbox_fc(feat)
        score_logit = self.score_fc(feat)
        return bbox_delta, score_logit
    
def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class TSPHead(nn.Module):
    def __init__(self,
                 n_classes=1,
                 in_channels=(128, 128, 128),
                 out_channels=128,
                 n_reg_outs=6,
                 voxel_size=.01,
                 pts_prune_threshold=(1200,4000),
                 top_pts_threshold=None,
                 volume_threshold=27,
                 r=(13,13),
                 assign_type='volume',
                 prune_threshold=(0.3,0.7),
                 random_prune_threshold=(1200,4000),
                 com_threshold = 0.15,
                 num_samples_com=2400,
                 seg_thr = 0.3,
                 train_cfg=None,
                 test_cfg=dict(nms_pre=1, iou_thr=.5, score_thr=.01),
                 keep_loss_weight = 1.0,
                 bbox_loss_weight = 1.0,
                 seg_loss_weight = 2.,
                 seg_loss_dice_weight = .1,
                 use_seg = False,
                 use_seg_external_self_attn=False,
                 use_external_attn_bi_layer=(),
                 use_text_guided_external_attn_bi_layer=(),
                 use_film_text_guided_external_attn_bi_layer=(),
                 external_attn_coef=4,
                 external_attn_k=None,
                 top_pts_threshold_det=None,
                 measure_fps_detail=False):
        super(TSPHead, self).__init__()
        if external_attn_k is None:
            external_attn_k = {}
        self.external_attn_k = external_attn_k
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.assign_type = assign_type
        self.volume_threshold = volume_threshold
        self.r = r
        self.prune_threshold = prune_threshold
        self.keep_loss_weight = keep_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.seg_loss_dice_weight = seg_loss_dice_weight
        self.use_seg = use_seg
        self.use_seg_external_self_attn = use_seg_external_self_attn
        self.use_external_attn_bi_layer = set(use_external_attn_bi_layer)
        self.use_text_guided_external_attn_bi_layer = set(use_text_guided_external_attn_bi_layer)
        self.use_film_text_guided_external_attn_bi_layer = set(use_film_text_guided_external_attn_bi_layer)
        self.external_attn_coef = external_attn_coef
        self.measure_fps_detail = bool(measure_fps_detail)
        if top_pts_threshold is None:
            top_pts_threshold = 24 if use_seg else 32
        if top_pts_threshold_det is None:
            top_pts_threshold_det = 8 if use_seg else 32
        self.assigner = TR3DAssigner(
            top_pts_threshold=top_pts_threshold,
            top_pts_threshold_det=top_pts_threshold_det,
            label2level=[0]
        )
        self.bbox_loss = AxisAlignedIoULoss2(mode='diou', reduction='none')
        self.cls_loss = FocalLoss(reduction='none')
        self.com_loss = FocalLoss(reduction='none')
        self.keep_loss = FocalLoss(reduction='mean', use_sigmoid=True)
        self.seg_loss = FocalLoss(reduction='mean', use_sigmoid=True)
        self.seg_loss_dice = DiceLoss(reduction='mean', use_sigmoid=True)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_samples = (3200,320)
        self.num_samples_com = num_samples_com
        self.com_threshold = com_threshold
        self.random_prune_threshold = random_prune_threshold
        self.seg_thr = seg_thr
        
        self.padding = 0.08
        self.min_pts_threshold = 16
        self.max_seg_bbox = 36
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)

    def _profile_begin(self):
        if self.measure_fps_detail and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return ("cuda", start_event, end_event)
        return ("cpu", time.perf_counter())

    def _profile_end(self, handle):
        if handle[0] == "cuda":
            _, start_event, end_event = handle
            end_event.record()
            end_event.synchronize()
            return float(start_event.elapsed_time(end_event)) / 1000.0
        return float(time.perf_counter() - handle[1])


    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_down_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_up_block(in_channels, out_channels, generative=False):
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.keep_conv = nn.ModuleList([
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3)
        ])
        self.pos_embed = PositionEmbeddingLearned(3, 128)
        bi_layer0 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False,
            use_external_attn=0 in self.use_external_attn_bi_layer,
            use_text_guided_external_attn=0 in self.use_text_guided_external_attn_bi_layer,
            use_film_text_guided_external_attn=0 in self.use_film_text_guided_external_attn_bi_layer,
            external_attn_coef=self.external_attn_coef,
            external_attn_k=self.external_attn_k.get('bi_layer0')
        )
        bi_layer1 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False,
            use_external_attn=1 in self.use_external_attn_bi_layer,
            use_text_guided_external_attn=1 in self.use_text_guided_external_attn_bi_layer,
            use_film_text_guided_external_attn=1 in self.use_film_text_guided_external_attn_bi_layer,
            external_attn_coef=self.external_attn_coef,
            external_attn_k=self.external_attn_k.get('bi_layer1')
        )
        bi_layer2 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False,
            use_external_attn=2 in self.use_external_attn_bi_layer,
            use_text_guided_external_attn=2 in self.use_text_guided_external_attn_bi_layer,
            use_film_text_guided_external_attn=2 in self.use_film_text_guided_external_attn_bi_layer,
            external_attn_coef=self.external_attn_coef,
            external_attn_k=self.external_attn_k.get('com_trans')
        )
        # pdb.set_trace()
        self.keep_trans = nn.ModuleList([BiEncoder(bi_layer0, 2), BiEncoder(bi_layer1, 2)])
        self.com_trans = BiEncoder(bi_layer2, 2)
        self.pruning = ME.MinkowskiPruning()
        self.com_cls = nn.Conv1d(128, 1, kernel_size=1, bias=True)


        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            self.__setattr__(
                        f'lateral_block_{i}',
                        self.make_block(in_channels[i], in_channels[i]))
            if i == 0:
                self.__setattr__(
                    f'out_block_{i}',
                    self.make_block(in_channels[i], out_channels))

        self.fuse = MinkowskiFeatureFusionBlock(128, 128, 128)

        if self.use_seg and self.use_seg_external_self_attn:
            # External self-attention before each seg upsample stage.
            self.seg_pos_embed_128 = PositionEmbeddingLearned(3, 128)
            self.seg_pos_embed_64 = PositionEmbeddingLearned(3, 64)
            self.seg_text_proj_64 = nn.Linear(128, 64)
            self.seg_self_attn_128 = PosTransformerEncoderLayerNoFFN(
                d_model=128,
                nhead=8,
                dropout=0.1,
                use_external_attn=True,
                use_text_guided_external_attn=True,
                use_film_text_guided_external_attn=True,
                external_attn_coef=self.external_attn_coef,
                external_attn_k=self.external_attn_k.get('seg_128'))
            self.seg_self_attn_64 = PosTransformerEncoderLayerNoFFN(
                d_model=64,
                nhead=8,
                dropout=0.1,
                use_external_attn=True,
                use_text_guided_external_attn=True,
                use_film_text_guided_external_attn=True,
                external_attn_coef=self.external_attn_coef,
                external_attn_k=self.external_attn_k.get('seg_64'))

        if self.use_seg:

            # pdb.set_trace()
            self.upsample_st_4 = nn.Sequential(
                            ME.MinkowskiConvolutionTranspose(
                                64,
                                64,
                                kernel_size=3,
                                stride=4,
                                dimension=3),
                            ME.MinkowskiBatchNorm(64),
                            ME.MinkowskiReLU(inplace=True))      
            self.upsample_st_2 = nn.Sequential(
                            ME.MinkowskiConvolutionTranspose(
                                128,
                                64,
                                kernel_size=3,
                                stride=2,
                                dimension=3),
                            ME.MinkowskiBatchNorm(64),
                            ME.MinkowskiReLU(inplace=True)) 
            self.conv_32_ch = nn.Sequential(
                            ME.MinkowskiConvolution(
                                64,
                                32,
                                kernel_size=3,
                                stride=1,
                                dimension=3),
                            ME.MinkowskiBatchNorm(32),
                            ME.MinkowskiReLU(inplace=True))  
            self.seg_unet = MinkUNet14B(in_channels=32, out_channels=1, D=3)

            # ---- refine head (uses feats BEFORE seg_unet) ----
            self.use_refine = True
            self.refine_grid_size = 14
            self.refine_pool = RoIAwarePool3d(
                out_size=self.refine_grid_size,
                max_pts_each_voxel=128
            )
            self.refine_head = SimpleRefineHead(in_ch=32, grid_size=self.refine_grid_size, hidden=256)

            self.refine_score_loss = nn.BCEWithLogitsLoss(reduction='mean')
            
            self.refine_score_loss_weight = 0.0     #1.0
            self.refine_bbox_loss_weight = 100.0  #100.


    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        for i in range(len(self.keep_conv)):
            nn.init.normal_(self.keep_conv[i].kernel, std=.01)

        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n) and ('seg_unet' not in n) \
                and ('keep_conv' not in n) and ('loss' not in n) :
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)       


    def _seg_external_self_attn(self,
                                x: ME.SparseTensor,
                                attn_layer,
                                pos_embed_layer,
                                text_feats=None,
                                text_attention_mask=None,
                                text_proj=None):
        """Apply external-attention self-attention on sparse voxels per batch."""
        if x is None or x.features.shape[0] == 0:
            return x

        sampled_coords, sampled_features = [], []
        max_len = max(len(permutation) for permutation in x.decomposition_permutations)

        for permutation in x.decomposition_permutations:
            if len(permutation) < max_len:
                padding_size = max_len - len(permutation)
                padded_features = torch.cat(
                    [x.features[permutation],
                     torch.zeros((padding_size, x.features[permutation].shape[1]),
                                 dtype=x.features.dtype,
                                 device=x.device)],
                    dim=0)
                padded_coords = torch.cat(
                    [x.coordinates[permutation],
                     -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                 dtype=x.coordinates.dtype,
                                 device=x.device)],
                    dim=0)
                sampled_features.append(padded_features)
                sampled_coords.append(padded_coords)
            else:
                sampled_features.append(x.features[permutation])
                sampled_coords.append(x.coordinates[permutation])

        sampled_features = torch.stack(sampled_features)
        sampled_coords = torch.stack(sampled_coords)
        padding_mask = sampled_coords[:, :, 0] == -1
        pos_feats = pos_embed_layer(
            sampled_coords[:, :, 1:] * self.voxel_size).transpose(1, 2).contiguous()

        text_global = None
        if text_feats is not None:
            text_for_attn = text_proj(text_feats) if text_proj is not None else text_feats
            if text_attention_mask is not None:
                valid_mask = (~text_attention_mask).unsqueeze(-1).type_as(text_for_attn)
                denom = valid_mask.sum(dim=1).clamp(min=1.0)
                text_global = (text_for_attn * valid_mask).sum(dim=1) / denom
            else:
                text_global = text_for_attn.mean(dim=1)

        sampled_features = attn_layer(
            sampled_features.transpose(0, 1).contiguous(),
            pos_feats.transpose(0, 1).contiguous(),
            src_key_padding_mask=padding_mask,
            text_feat=text_global).transpose(0, 1).contiguous()

        valid_mask = ~padding_mask
        sampled_features = sampled_features[valid_mask]
        sampled_coords = sampled_coords[valid_mask]

        return ME.SparseTensor(
            features=sampled_features,
            coordinates=sampled_coords,
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
            device=x.device)
    

    def _forward_single(self, x: ME.SparseTensor):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6].clamp(min=-10.0, max=10.0))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        scores = self.cls_conv(x)
        cls_pred = scores.features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:]* self.voxel_size)
        return bbox_preds, cls_preds, points
    
    def forward(self, x_all, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas, pc=None):
        bboxes_level = []
        bboxes_state = []
        if self.assign_type == 'volume':
            for idx in range(len(img_metas)):
 
                bbox_all = gt_all_bbox_new[idx]
                bbox_level = torch.ones([bbox_all.shape[0], 1])
                bbox_state_all = torch.cat((bbox_level, bbox_all.gravity_center, bbox_all.tensor[:, 3:]), dim=1)

                bbox_gt = gt_bboxes[idx]
                bbox_state_gt = torch.cat((bbox_gt.gravity_center, bbox_gt.tensor[:, 3:]), dim=1)                
                bbox_auxi = auxi_bbox[idx]
                bbox_state_auxi = torch.cat((bbox_auxi.gravity_center, bbox_auxi.tensor[:, 3:]), dim=1)
                bbox_state_auxi_gt = torch.cat((bbox_state_gt, bbox_state_auxi), dim=0)
                bbox_level = torch.zeros([bbox_state_auxi_gt.shape[0], 1])
                bbox_state_auxi_gt = torch.cat((bbox_level, bbox_state_auxi_gt), dim=1)
                
                bbox_state = torch.cat((bbox_state_all, bbox_state_auxi_gt), dim=0)
                
                bboxes_level.append(bbox_state[:,[0]])
                bboxes_state.append(bbox_state)
        
        bbox_preds, cls_preds, points = [], [], []
        keep_gts = []
        keep_preds, prune_masks = [], []
        prune_mask = None
        # pdb.set_trace()
        inputs = x_all[2:]
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1): # 2,1,0
            if i ==1 :  #  1,0         
                prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas) 

                keep_gt = []
                for permutation in x.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                        coordinate_manager=x.coordinate_manager)
                x = x + x_level
                x = self._prune_training(x, prune_training_keep, i) 
            elif i == 0:
                prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas) 
                keep_gt = []
                for permutation in x.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                prune_threshold_ = np.random.randint(self.random_prune_threshold[0], self.random_prune_threshold[1])
                self.pts_prune_threshold = (prune_threshold_,self.pts_prune_threshold[1])
                x = self._prune_training(x, prune_training_keep, i)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                        coordinate_manager=x.coordinate_manager)
                x_ori = x + x_level
                
                
                sampled_coords,sampled_features, original_indices = [],[],[]
                
                for permutation in inputs[0].decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation))[:self.num_samples_com]
                        choice = torch.sort(choice).values
                        sampled_features.append(inputs[0].features[permutation][choice])
                        sampled_coords.append(inputs[0].coordinates[permutation][choice])
                    else:
                        padding_size = self.num_samples_com - len(permutation)      
                        padded_features = torch.cat(
                            [inputs[0].features[permutation], torch.zeros((padding_size, inputs[0].features[permutation].shape[1]), 
                                                                  dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0) 
                        padded_coords = torch.cat(
                            [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                     dtype=inputs[0].coordinates.dtype).to(inputs[0].device)], 
                                                                     dim=0)  
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                
                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :,0] != -1
                com_pred_training = [com_pred[k][valid_mask[k]] for k in range(len(com_pred))]
                com_coords_training = [sampled_coords[k][valid_mask[k]][:,1:]*self.voxel_size for k in range(len(com_pred))]
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > self.com_threshold
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]                
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]                   
                
                x_com_features = x.features_at_coordinates(sampled_coords.float())     
                x_com_features = x_com_features + sampled_features           
                x = ME.SparseTensor(features=torch.cat((x_ori.features,x_com_features),dim=0), 
                                    coordinates=torch.cat((x_ori.coordinates,sampled_coords),dim=0), 
                                    coordinate_manager=x_ori.coordinate_manager, tensor_stride=x_ori.tensor_stride, device=x_ori.device)
            if i > 0: # 2,1
                sampled_coords,sampled_features, original_indices = [],[],[]
                prune_mask = torch.zeros(x.shape[0], dtype=torch.bool).to(x.device)
                for permutation in x.decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples[i-1]:
                        choice = torch.randperm(len(permutation))[:self.num_samples[i-1]]
                        choice = torch.sort(choice).values
                        sampled_features.append(x.features[permutation][choice])
                        sampled_coords.append(x.coordinates[permutation][choice])
                        prune_mask[permutation[choice]] = True
                    else:
                        padding_size = self.num_samples[i-1] - len(permutation)      
                        padded_features = torch.cat(
                            [x.features[permutation], torch.zeros((padding_size, x.features[permutation].shape[1]), 
                                                                  dtype=x.features.dtype).to(x.device)], dim=0) 
                        padded_coords = torch.cat(
                            [x.coordinates[permutation], -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                                                     dtype=x.coordinates.dtype).to(x.device)], 
                                                                     dim=0)  
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                        prune_mask[permutation] = True
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.keep_trans[i-1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                
                x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords, 
                                    coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)
                keep_scores = self.keep_conv[i-1](x) # 1 MLP
                prune_training_keep = ME.SparseTensor(
                                    -keep_scores.features,
                                    coordinate_map_key=keep_scores.coordinate_map_key,
                                    coordinate_manager=keep_scores.coordinate_manager)
                
     
                keep_pred = keep_scores.features
                prune_inference = keep_pred
                keeps = []

                try:
                    for permutation in x.decomposition_permutations:
                        keeps.append(keep_pred[permutation])
                except:
                    pdb.set_trace()
                keep_preds.append(keeps)
                
            x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                out = self.__getattr__(f'out_block_{i}')(x)
        out = self.fuse(out, text_feats[:, 0])
        bbox_pred, cls_pred, point = self._forward_single(out)
        if self.use_seg:
            # pdb.set_trace()
            if self.use_seg_external_self_attn:
                x = self._seg_external_self_attn(
                    x,
                    self.seg_self_attn_128,
                    self.seg_pos_embed_128,
                    text_feats=text_feats,
                    text_attention_mask=text_attention_mask)
            x = self.upsample_st_2(x) + x_all[1]
            if self.use_seg_external_self_attn:
                x = self._seg_external_self_attn(
                    x,
                    self.seg_self_attn_64,
                    self.seg_pos_embed_64,
                    text_feats=text_feats,
                    text_attention_mask=text_attention_mask,
                    text_proj=self.seg_text_proj_64)
            x = self.upsample_st_4(x) + x_all[0]
            seg_feats = self.conv_32_ch(x)
        else:
            seg_feats = None
        return [bbox_pred], [cls_pred], [point], keep_preds[::-1], keep_gts[::-1], bboxes_level, com_pred_training, com_coords_training, \
            seg_feats
    

    def _prune_inference(self, x, scores, layer_id):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            prune_mask = scores.new_zeros(
                (len(scores)), dtype=torch.bool)
            threshold = self.prune_threshold[layer_id]
            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > threshold
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True

            kept = int(prune_mask.sum().item())
            if kept == 0 and scores.numel() > 0:
                logger.warning(
                    "Prune inference removed all points at layer %s (threshold=%s, scores_range=[%.6f, %.6f]); applying per-sample Top-1 fallback.",
                    layer_id,
                    threshold,
                    float(scores.min().item()),
                    float(scores.max().item()),
                )
                for permutation in x.decomposition_permutations:
                    if len(permutation) == 0:
                        continue
                    score = 1 - scores[permutation].sigmoid()
                    score = score.reshape([len(score)])
                    top_idx = torch.argmax(score)
                    prune_mask[permutation[top_idx]] = True
                kept = int(prune_mask.sum().item())

        if kept == 0:
            logger.warning(
                "Prune inference fallback failed at layer %s; keeping original tensor unchanged.",
                layer_id,
            )
            return x
        return self.pruning(x, prune_mask)


    def _prune_training(self, x, scores, layer_id):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold[layer_id])
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x


    @torch.no_grad()
    def _get_keep_voxel(self, input, cur_level, bboxes_state, input_metas):
        bboxes = []
        for size in range(len(input_metas)):
            bboxes.append([])
        for idx in range(len(input_metas)):
            for n in range(len(bboxes_state[idx])):
                if bboxes_state[idx][n][0] < (cur_level - 1):    
                    bboxes[idx].append(bboxes_state[idx][n])
        idx = 0
        mask = []
        l0 = self.voxel_size * 2 ** 2  # pool  True :2**3  False:2**2
        for idx, permutation in enumerate(input.decomposition_permutations):
            point = input.coordinates[permutation][:, 1:]* self.voxel_size
            if len(bboxes[idx]) != 0:
                point = input.coordinates[permutation][:, 1:]* self.voxel_size
                boxes = bboxes[idx]
                level = 3
                bboxes_level = [[] for _ in range(level)]
                for n in range(len(boxes)):
                    for l in range(level):
                        if boxes[n][0] == l:
                            bboxes_level[l].append(boxes[n])
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                for l in range(level):
                    if len(bboxes_level[l]) != 0:
                        point_l = point.unsqueeze(1).expand(len(point), len(bboxes_level[l]), 3)
                        boxes_l = torch.cat(bboxes_level[l]).reshape([-1, 8]).to(point.device)
                        boxes_l = boxes_l.expand(len(point), len(bboxes_level[l]), 8)
                        shift = torch.stack(
                            (point_l[..., 0] - boxes_l[..., 1], point_l[..., 1] - boxes_l[..., 2],
                            point_l[..., 2] - boxes_l[..., 3]),
                            dim=-1).permute(1, 0, 2)
                        shift = rotation_3d_in_axis(
                            shift, -boxes_l[0, :, 7], axis=2).permute(1, 0, 2)
                        centers = boxes_l[..., 1:4] + shift
                        up_level_l = self.r[cur_level-2] 
                        dx_min = centers[..., 0] - boxes_l[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dx_max = boxes_l[..., 1] - centers[..., 0] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2 
                        dy_min = centers[..., 1] - boxes_l[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dy_max = boxes_l[..., 2] - centers[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dz_min = centers[..., 2] - boxes_l[..., 3] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dz_max = boxes_l[..., 3] - centers[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2


                        distance = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
                        inside_box_condition = distance.min(dim=-1).values > 0
                        inside_box_condition = inside_box_condition.sum(dim=1)
                        inside_box_condition = inside_box_condition >= 1
                        inside_box_conditions += inside_box_condition
                mask.append(inside_box_conditions)
            else:
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                mask.append(inside_box_conditions)

        prune_mask = torch.cat(mask)
        prune_mask = prune_mask.to(input.device)
        return prune_mask
    

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)
            ).clamp(max=10.0))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)


    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta,
                     com_pred,com_coords,):
        # pdb.set_trace()
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)
        pos_bbox_preds_format = bbox_preds.new_zeros((0, bbox_preds.shape[1]))
        score = cls_preds.new_zeros((0, cls_preds.shape[1]))
        label = gt_labels.new_zeros((0,), dtype=gt_labels.dtype)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)

        cls_loss = self.cls_loss(cls_preds, cls_targets)
        
        assigned_ids_com = self.assigner.assign([com_coords], gt_bboxes, gt_labels, img_meta)
        # cls loss
        pos_mask_com = assigned_ids_com >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask_com, gt_labels[assigned_ids_com], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask_com),), n_classes)

        com_loss = self.com_loss(com_pred, cls_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        # pdb.set_trace()
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            
            pos_bbox_preds_format = self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(pos_bbox_preds_format),
                self._bbox_to_loss(pos_bbox_targets))     
                
            score = cls_preds[pos_mask]
            label = gt_labels[assigned_ids][pos_mask]   
        else:
            bbox_loss = None
            
        # pdb.set_trace()
        return bbox_loss, cls_loss, pos_mask, com_loss, pos_mask_com, pos_bbox_preds_format, score, label


    def _loss(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas, 
              keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training, gt_points, targets, seg_feats):
        bbox_losses, cls_losses, pos_masks, com_losses, pos_masks_com, selected_bboxes, selected_scores, selected_labels \
            = [], [], [], [], [], [], [], []

        #keep loss
        keep_losses = 0
        for i in range(len(img_metas)):
            k_loss = 0
            keep_pred = [x[i] for x in keep_preds]
            keep_gt = [x[i] for x in keep_gts]
            for j in range(len(keep_preds)):
                pred = keep_pred[j]
                gt = (keep_gt[j]).long()

                if gt.sum() != 0:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=gt.sum())
                    k_loss = torch.mean(keep_loss) / 3 + k_loss
                else:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=len(gt))  
                    k_loss = torch.mean(keep_loss) / 3 + k_loss

            keep_losses = keep_losses + k_loss

        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask, com_loss,pos_mask_com, selected_bbox,selected_score, selected_label = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                com_pred = com_pred_training[i],
                com_coords = com_coords_training[i],)
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            com_losses.append(com_loss)
            pos_masks.append(pos_mask)
            pos_masks_com.append(pos_mask_com)

            if len(selected_bbox)>self.max_seg_bbox:
                indices = torch.randperm(selected_bbox.shape[0])[:self.max_seg_bbox]
                selected_bbox = selected_bbox[indices]
                selected_score = selected_score[indices]
                selected_label = selected_label[indices]
            selected_bboxes.append(selected_bbox)
            selected_scores.append(selected_score)
            selected_labels.append(selected_label)
            
        # pdb.set_trace()
        if self.use_seg:
            seg_preds, targets, v2r, r2scene, rois, scores, gt_idxs, refine_delta, refine_score = \
                self._forward_seg(seg_feats, targets, selected_bboxes, selected_scores, selected_labels)
            seg_loss, seg_loss_dice = self._loss_second(seg_preds, targets, v2r, r2scene, rois, gt_idxs,gt_bboxes, gt_labels, img_metas)
            refine_bbox_loss, refine_score_loss = self._loss_refine(
                rois_list=rois,
                refine_delta=refine_delta,
                refine_score_logit=refine_score,
                gt_bboxes=gt_bboxes,
                img_metas=img_metas
            )
        # pdb.set_trace()
        loss_dict = dict(
            bbox_loss=self.bbox_loss_weight * (
                torch.mean(torch.cat(bbox_losses))
                if len(bbox_losses) > 0 else torch.sum(torch.cat(cls_losses)) * 0.0
            ),
            cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)).clamp(min=1).float(),
            keep_loss=self.keep_loss_weight * keep_losses / len(img_metas),
            com_loss=torch.sum(torch.cat(com_losses)) / torch.sum(torch.cat(pos_masks_com)).clamp(min=1).float(),
        )
        # pdb.set_trace()
        if self.use_seg:
            loss_dict.update(dict(
                seg_loss=self.seg_loss_weight * seg_loss,
                seg_loss_dice=self.seg_loss_dice_weight * seg_loss_dice,
                refine_bbox_loss=self.refine_bbox_loss_weight * refine_bbox_loss,
                refine_score_loss=self.refine_score_loss_weight * refine_score_loss,
            ))
        return loss_dict
        # return dict(refine_bbox_loss=self.refine_bbox_loss_weight * refine_bbox_loss,refine_score_loss=self.refine_score_loss_weight * refine_score_loss )
    
    def _loss_refine(self, rois_list, refine_delta, refine_score_logit, gt_bboxes, img_metas):
        if refine_delta is None or refine_delta.numel() == 0:
            zero = refine_score_logit.sum() * 0.0 if refine_score_logit is not None else torch.tensor(0.0, device=rois_list[0].device)
            return zero, zero

        rois_all = torch.cat([r[:, :6] for r in rois_list], dim=0).to(refine_delta.device).detach()
        pred_boxes = self.decode_roi_residual(rois_all, refine_delta)

        lens = [len(r) for r in rois_list]
        scene_ids = torch.cat([
            torch.full((n,), i, device=refine_delta.device, dtype=torch.long)
            for i, n in enumerate(lens)
        ], dim=0)

        reg_losses = []
        score_losses = []

        reg_loss_fn = nn.SmoothL1Loss(reduction='none')

        for i in range(len(img_metas)):
            mask = scene_ids == i
            if mask.sum() == 0:
                continue

            cur_rois = rois_all[mask]
            cur_pred_boxes = pred_boxes[mask]
            cur_pred_delta = refine_delta[mask]
            cur_score_logit = refine_score_logit[mask]

            gt = gt_bboxes[i]
            if len(gt) == 0:
                q = cur_pred_boxes.new_zeros((cur_pred_boxes.shape[0], 1))
                score_losses.append(self.refine_score_loss(cur_score_logit, q))
                continue

            gt6 = torch.cat((gt.gravity_center, gt.tensor[:, 3:6]), dim=1).to(cur_pred_boxes.device)

            # assign gt by ROI, not by pred_boxes
            roi_ious = axis_aligned_iou_3d(cur_rois, gt6)
            best_iou_roi, best_id = roi_ious.max(dim=1)

            matched_gt = gt6[best_id]
            reg_target = self.encode_roi_residual(cur_rois, matched_gt)

            reg_loss = reg_loss_fn(cur_pred_delta, reg_target).mean(dim=1)
            reg_losses.append(reg_loss)

            pred_ious = axis_aligned_iou_3d(cur_pred_boxes, gt6)
            best_iou_pred, _ = pred_ious.max(dim=1)
            q = iou_guided_quality(best_iou_pred).unsqueeze(1)

            score_losses.append(self.refine_score_loss(cur_score_logit, q))

        if len(reg_losses) == 0:
            reg_loss = refine_delta.sum() * 0.0
        else:
            reg_loss = torch.cat(reg_losses).mean()

        if len(score_losses) == 0:
            score_loss = refine_score_logit.sum() * 0.0
        else:
            score_loss = torch.stack(score_losses).mean()

        return reg_loss, score_loss

    def _loss_second(self, cls_preds, targets, v2r, r2scene, rois, gt_idxs,
                    gt_bboxes, gt_labels, img_metas):
        # pdb.set_trace()
        try:
            v2scene = r2scene[v2r]
            seg_losses = []
            seg_losses_dice = []
            for i in range(len(img_metas)):
                seg_loss, seg_loss_dice = self._loss_second_single(
                    cls_preds=cls_preds[v2scene == i],
                    targets=targets[v2scene == i],
                    v2r=v2r[v2scene == i],
                    rois=rois[i],
                    gt_idxs=gt_idxs[i],
                    gt_bboxes=gt_bboxes[i],
                    gt_labels=gt_labels[i],
                    img_meta=img_metas[i],
                )
                seg_losses.append(seg_loss)
                seg_losses_dice.append(seg_loss_dice)

            # pdb.set_trace()
            return torch.mean(torch.stack(seg_losses)), torch.mean(torch.stack(seg_losses_dice))
        except Exception as e:
            print(e)
            pdb.set_trace()
    
    def _loss_second_single(self, cls_preds, targets, v2r, rois, gt_idxs, gt_bboxes, gt_labels, img_meta):
        if len(rois) == 0 or cls_preds.shape[0] == 0:
            zero_loss = cls_preds.sum().float() * 0.
            return zero_loss, zero_loss, zero_loss, zero_loss, zero_loss
        v2r = v2r - v2r.min()
        
        assert len(torch.unique(v2r)) == len(rois)
        assert torch.all(torch.unique(v2r) == torch.arange(0, v2r.max() + 1).to(v2r.device))
        assert torch.max(gt_idxs) < len(gt_bboxes)

        v2bbox = gt_idxs[v2r.long()]
        assert torch.unique(v2bbox)[0] != -1
        # inst_targets = targets[:, 0]
        # pdb.set_trace()
        # seg_targets = targets[:, 1]

        # seg_preds = cls_preds[:, :-1]
        # inst_preds = cls_preds[:, -1]

        # labels = v2bbox == inst_targets

        # seg_targets[seg_targets == -1] = self.n_classes
        # pdb.set_trace()
        seg_loss = self.seg_loss(cls_preds, (targets).long())
        seg_loss_dice = self.seg_loss_dice(cls_preds, (targets).long())
            
        return seg_loss, seg_loss_dice
    
    def forward_train(self, x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, \
        gt_points, targets, img_metas,pc=None):
        
        bbox_preds, cls_preds, points, keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training, seg_feats = \
            self(x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc)
        # pdb.set_trace()

        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level,
                          com_pred_training, com_coords_training, gt_points, targets, seg_feats)
        
    def _forward_seg(self, x, targets, rois, scores, labels):
        # rois = [b[0] for b in bbox_list]
        # scores = [b[1] for b in bbox_list]
        # labels = [b[2] for b in bbox_list]
        # levels = [torch.zeros(len(b[0])) for b in bbox_list]
        # pdb.set_trace()
        feats_with_targets = ME.SparseTensor(torch.cat((x.features, targets), dim=1), x.coordinates)
        # pdb.set_trace()
        tensors, ids, rois, scores, labels = self.extract(feats_with_targets, rois, scores, labels)        
        # pdb.set_trace()
        
        if tensors.features.shape[0] == 0:
            return (targets.new_zeros((0, 1)),
                    targets.new_zeros((0, 1)),
                    targets.new_zeros(0),
                    targets.new_zeros(0),
                    [targets.new_zeros((0, 7)) for i in range(len(rois))],
                    [targets.new_zeros(0) for i in range(len(rois))],
                    [targets.new_zeros(0) for i in range(len(rois))],
                    None)

        feats = ME.SparseTensor(tensors.features[:, :-1], tensors.coordinates)
        targets = tensors.features[:, -1:]

        # seg
        if self.use_seg:
            preds = self.seg_unet(feats).features
        else:
            preds = None

        # ---- refine (uses feats BEFORE seg_unet) ----
        refine_bbox_delta = None
        refine_score_logit = None
        if getattr(self, "use_refine", False) and feats.features.shape[0] > 0:
            # ---- refine (uses feats BEFORE seg_unet) ----
            refine_bbox_delta = None
            refine_score_logit = None
            if getattr(self, "use_refine", False):
                R = sum(len(r) for r in rois)
                if R > 0:
                    # pdb.set_trace()
                    pooled_feats = self.roi_grid_pool_axis_aligned_cuda(feats, rois, grid_size=self.refine_grid_size)  # (R, g, g, g, C)

                    refine_bbox_delta, refine_score_logit = self.refine_head(pooled_feats)

        return preds, targets, feats.coordinates[:, 0].long(), ids, rois, scores, labels, refine_bbox_delta, refine_score_logit
    
    def extract(self, tensors, rois, scores, labels):
        # pdb.set_trace()
        # for level, x in enumerate(tensors):
        n_rois = 0
        new_coordinates, new_features, new_roi, new_score, new_label, ids = [], [], [], [], [], []
        for i, (coordinates, features) in enumerate(zip(*tensors.decomposed_coordinates_and_features)):        
            roi = rois[i]
            roi = torch.cat((
                roi[:,  :3],
                roi[:,  3:6] + self.padding,
                roi[:, 6:]), dim=1)
            score = scores[i]
            label = labels[i]
            new_index, new_coordinate, new_feature, roi, score, label = self._extract_single(
                coordinates, features, roi, score, label)  
            new_index = new_index + n_rois
            n_rois += len(roi)
            new_coordinate = torch.cat((
                new_index.unsqueeze(1), new_coordinate), dim=1)
            new_coordinates.append(new_coordinate)
            new_features.append(new_feature)
            ids += [i] * len(roi)
            roi = torch.cat((roi[:, :3],
                        roi[:,  3:6] - self.padding,
                        roi[:, 6:]), dim=1)
            new_roi.append(roi)
            new_score.append(score)
            new_label.append(label)
            
        new_coords = torch.cat(new_coordinates).int().contiguous()
        new_feats = torch.cat(new_features).contiguous()

        new_tensors = ME.SparseTensor(
            features=new_feats,
            coordinates=new_coords,
            tensor_stride=tensors.tensor_stride,
            device=new_feats.device
)
        # pdb.set_trace()
        new_ids = tensors.coordinates.new_tensor(ids)
        new_rois = new_roi
        new_scores = new_score
        new_labels = new_label
        return new_tensors, new_ids, new_rois, new_scores, new_labels    
    
    def _extract_single(self, coordinates, features, rois, scores, labels):
        # coordinates: of shape (n_points, 3)
        # features: of shape (n_points, c)
        # voxel_size: float
        # rois: of shape (n_rois, 7)
        # -> new indices of shape n_new_points
        # -> new coordinates of shape (n_new_points, 3)
        # -> new features of shape (n_new_points, c + 3)
        # -> new rois of shape (n_new_rois, 7)
        # -> new scores of shape (n_new_rois)
        # -> new labels of shape (n_new_rois)
        n_points = len(coordinates)
        n_boxes = len(rois)
        if n_boxes == 0:
            return (coordinates.new_zeros(0),
                    coordinates.new_zeros((0, 3)),
                    features.new_zeros((0, features.shape[1])),
                    features.new_zeros((0, 7)),
                    features.new_zeros(0),
                    coordinates.new_zeros(0))
        points = coordinates * self.voxel_size
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        # pdb.set_trace()
        rois = torch.cat([rois, torch.zeros((rois.size(0), 1), device=rois.device)], dim=1)
        rois = rois.unsqueeze(0).expand(n_points, n_boxes, 7)
        face_distances = get_face_distances(points, rois)
        inside_condition = face_distances.min(dim=-1).values > 0
        # pdb.set_trace()
        # if np.random.randint(1, 11)>3:
        #     self.min_pts_threshold = 16
        # else:
        #     self.min_pts_threshold = 160000
        min_pts_condition = inside_condition.sum(dim=0) > self.min_pts_threshold
        inside_condition = inside_condition[:, min_pts_condition]
        rois = rois[0, min_pts_condition]
        scores = scores[min_pts_condition]
        labels = labels[min_pts_condition]
        nonzero = torch.nonzero(inside_condition)
        new_coordinates = coordinates[nonzero[:, 0]]
        
        return nonzero[:, 1], new_coordinates, features[nonzero[:, 0]], rois, scores, labels
    
    def encode_roi_residual(self, rois, gt_boxes):
        """
        rois: (N,6) [cx,cy,cz,w,h,l]
        gt_boxes: (N,6)
        return:
            delta: (N,6)
        """
        roi_ctr = rois[:, :3]
        roi_size = rois[:, 3:6].clamp(min=1e-6)

        gt_ctr = gt_boxes[:, :3]
        gt_size = gt_boxes[:, 3:6].clamp(min=1e-6)

        delta_ctr = (gt_ctr - roi_ctr) / roi_size
        delta_size = torch.log(gt_size / roi_size)

        # delta_ctr = gt_ctr - roi_ctr
        # delta_size = torch.log(gt_size)

        return torch.cat([delta_ctr, delta_size], dim=1)
    
    def decode_roi_residual(self, rois, delta):
        roi_ctr = rois[:, :3]
        roi_size = rois[:, 3:6].clamp(min=1e-6)

        pred_ctr = roi_ctr + delta[:, :3] * roi_size
        pred_size = roi_size * torch.exp(delta[:, 3:6].clamp(min=-10.0, max=10.0))

        # pred_ctr = roi_ctr + delta[:, :3] 
        # pred_size = torch.exp(delta[:, 3:6])

        return torch.cat([pred_ctr, pred_size], dim=1)

    def roi_grid_pool_axis_aligned_cuda(self, feats: ME.SparseTensor, rois_list, grid_size):
        """
        feats:
            coordinates: (N, 4) [roi_idx, x, y, z]
            features:    (N, C)
        rois_list:
            list of per-scene rois, each (n_i, 6)
        return:
            pooled_feats: (R, g, g, g, C)
        """
        if feats.features.shape[0] == 0:
            C = feats.features.shape[1]
            R = sum(len(r) for r in rois_list)
            return feats.features.new_zeros((R, grid_size, grid_size, grid_size, C))

        device = feats.features.device
        rois_all = torch.cat([r[:, :6] for r in rois_list], dim=0).to(device)  # (R, 6)
        R = rois_all.shape[0]

        # axis-aligned -> heading = 0
        heading = rois_all.new_zeros((R, 1))
        rois7 = torch.cat([rois_all, heading], dim=1).contiguous()  # (R, 7)

        # coords: [roi_idx, x, y, z]
        roi_idx = feats.coordinates[:, 0].long()
        pts_world = feats.coordinates[:, 1:4].float() * self.voxel_size
        pts_feat = feats.features.contiguous()

        pooled_list = []
        C = pts_feat.shape[1]

        for rid in range(R):
            mask = (roi_idx == rid)
            if mask.sum() == 0:
                pooled = pts_feat.new_zeros((1, grid_size, grid_size, grid_size, C))
            else:
                cur_pts = pts_world[mask].contiguous()
                cur_feat = pts_feat[mask].contiguous()
                cur_roi = rois7[rid:rid + 1].contiguous()
                pooled = self.refine_pool(cur_roi, cur_pts, cur_feat, pool_method='max')
            pooled_list.append(pooled)

        pooled_feats = torch.cat(pooled_list, dim=0)  # (R, g, g, g, C)
        return pooled_feats

       
    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg['score_thr']
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg['iou_thr'])
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg['nms_pre'] > 0:
            _, ids = max_scores.topk(self.test_cfg['nms_pre'])
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        labels = boxes.new_zeros((1, ),dtype=int)
        boxes = img_meta['box_type_3d'](boxes, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        return boxes, scores, labels


    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results


    def forward_test(self, x_all, text_feats, text_attention_mask, targets, inverse_mapping, img_metas, pc=None, gt_bboxes=None):
        inputs = x_all[2:]
        x = inputs[-1]
        bbox_preds, cls_preds, points = [], [], []
        keep_scores = None
        self.last_stage_profile = {}
        self.last_external_attn_profile = {
            'bi_layer0': 0.0,
            'bi_layer1': 0.0,
            'bi_layer2': 0.0,
            'total': 0.0,
        }
        
        for i in range(len(inputs) - 1, -1, -1):
            if i ==1:
                if self.measure_fps_detail:
                    logger.info("Forward test layer %s: x points=%s", i, int(x.features.shape[0]))
                stage_handle = self._profile_begin()
                x = self._prune_inference(x, prune_inference, i)
                self.last_stage_profile[f'prune_layer{i}'] = self._profile_end(stage_handle)
                
                if x != None:
                    if self.measure_fps_detail:
                        logger.info("After prune layer %s: x points=%s", i, int(x.features.shape[0]))
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                else:
                    logger.warning("Forward test stopped at layer %s: x is None after pruning.", i)
                    break
            elif i ==0:
                if self.measure_fps_detail:
                    logger.info("Forward test layer %s: x points=%s", i, int(x.features.shape[0]))
                stage_handle = self._profile_begin()
                x = self._prune_inference(x, prune_inference, i)
                self.last_stage_profile[f'prune_layer{i}'] = self._profile_end(stage_handle)
                
                if x != None:
                    if self.measure_fps_detail:
                        logger.info("After prune layer %s: x points=%s", i, int(x.features.shape[0]))
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x_ori = x + x_level
                else:
                    logger.warning("Forward test stopped at layer %s: x is None after pruning.", i)
                    break
        
                sampled_coords,sampled_features, original_indices = [],[],[]
                
                for permutation in inputs[0].decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation))[:self.num_samples_com]
                        choice = torch.sort(choice).values
                        sampled_features.append(inputs[0].features[permutation][choice])
                        sampled_coords.append(inputs[0].coordinates[permutation][choice])
                    else:
                        padding_size = self.num_samples_com - len(permutation)      
                        padded_features = torch.cat(
                            [inputs[0].features[permutation], torch.zeros((padding_size, inputs[0].features[permutation].shape[1]), 
                                                                  dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0) 
                        padded_coords = torch.cat(
                            [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                     dtype=inputs[0].coordinates.dtype).to(inputs[0].device)], 
                                                                     dim=0)  
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                ext_handle = self._profile_begin()
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                ext_dt = self._profile_end(ext_handle)
                self.last_stage_profile['com_trans'] = ext_dt
                self.last_external_attn_profile['bi_layer2'] += ext_dt
                self.last_external_attn_profile['total'] += ext_dt
                
                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > self.com_threshold
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]                
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]                   
                
                x_com_features = x.features_at_coordinates(sampled_coords.float())     
                x_com_features = x_com_features + sampled_features           
                x = ME.SparseTensor(features=torch.cat((x_ori.features,x_com_features),dim=0), 
                                    coordinates=torch.cat((x_ori.coordinates,sampled_coords),dim=0), 
                                    coordinate_manager=x_ori.coordinate_manager, tensor_stride=x_ori.tensor_stride, device=x_ori.device)
                
            if i > 0:
                sampled_coords,sampled_features = [],[]
                len_x = []
                for permutation in x.decomposition_permutations:
                    len_x.append(len(x.coordinates[permutation]))
                # max_len_x = self.num_samples[i-1]
                max_len_x = int(torch.tensor(len_x).max())
                if len(len_x)>1:
                    for permutation in x.decomposition_permutations:
                        if len(permutation) > max_len_x:
                            choice = torch.randperm(len(permutation))[:max_len_x]
                            choice = torch.sort(choice).values
                            sampled_features.append(x.features[permutation][choice])
                            sampled_coords.append(x.coordinates[permutation][choice])
                        else:
                            padding_size = max_len_x - len(permutation)      
                            padded_features = torch.cat(
                                [x.features[permutation], torch.zeros((padding_size, x.features[permutation].shape[1]), 
                                                                    dtype=x.features.dtype).to(x.device)], dim=0) 
                            padded_coords = torch.cat(
                                [x.coordinates[permutation], -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                                                        dtype=x.coordinates.dtype).to(x.device)], 
                                                                        dim=0)   
                            sampled_features.append(padded_features)
                            sampled_coords.append(padded_coords)
                else:
                    for permutation in x.decomposition_permutations:
                        sampled_features.append(x.features[permutation])
                        sampled_coords.append(x.coordinates[permutation])                        
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                ext_handle = self._profile_begin()
                sampled_features, text_feats = self.keep_trans[i-1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                ext_dt = self._profile_end(ext_handle)
                self.last_stage_profile[f'keep_trans_layer{i-1}'] = ext_dt
                self.last_external_attn_profile[f'bi_layer{i-1}'] += ext_dt
                self.last_external_attn_profile['total'] += ext_dt
                
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords, 
                                    coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)
                keep_scores = self.keep_conv[i-1](x)
                keep_pred = keep_scores.features
                prune_inference = keep_pred

            x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                out = self.__getattr__(f'out_block_{i}')(x)
        start_time = time.perf_counter()
        out = self.fuse(out, text_feats[:, 0])
        bbox_pred, cls_pred, point = self._forward_single(out)
        results = self._get_bboxes([bbox_pred], [cls_pred], [point], img_metas)
        
        if self.use_seg:
            if self.use_seg_external_self_attn:
                x = self._seg_external_self_attn(
                    x,
                    self.seg_self_attn_128,
                    self.seg_pos_embed_128,
                    text_feats=text_feats,
                    text_attention_mask=text_attention_mask)
            x = self.upsample_st_2(x) + x_all[1]
            if self.use_seg_external_self_attn:
                x = self._seg_external_self_attn(
                    x,
                    self.seg_self_attn_64,
                    self.seg_pos_embed_64,
                    text_feats=text_feats,
                    text_attention_mask=text_attention_mask,
                    text_proj=self.seg_text_proj_64)
            x = self.upsample_st_4(x) + x_all[0]
            seg_feats = self.conv_32_ch(x)

            selected_bboxes,selected_scores,selected_labels = [],[],[]
            for box, score, label in results:
                box = torch.cat((box.gravity_center, box.tensor[:,  3:6]), dim=1)  
                selected_bboxes.append(box)
                selected_scores.append(score)
                selected_labels.append(label)

            src_idxs = torch.arange(0, x_all[0].features.shape[0]).to(inverse_mapping.device)
            # src_idxs = src_idxs.unsqueeze(1).expand(src_idxs.shape[0], 2)
            seg_preds, idxs, v2r, r2scene, rois, scores, gt_idxs, refine_delta, refine_score = \
                self._forward_seg(seg_feats, src_idxs.unsqueeze(-1), selected_bboxes, selected_scores, selected_labels)
            # seg_preds, targets_new, v2r, r2scene, rois, scores, gt_idxs = self._forward_seg(seg_feats, targets, selected_bboxes,selected_scores,selected_labels)

            # ---- apply refine to detection results ----
            if getattr(self, "use_refine", False) and refine_delta is not None and refine_delta.numel() > 0:

                new_results = []
                roi_offset = 0

                for i, (box3d, score_mat, label) in enumerate(results):
                    roi_count = len(rois[i])
                    if roi_count == 0:
                        new_results.append((box3d, score_mat, label))
                        continue

                    # Current scene is expected to have a single ROI when nms_pre=1;
                    # use the first ROI and its matching refine prediction.
                    roi = rois[i][0, :6].to(refine_delta.device)

                    # roi_center = roi[:3]
                    # delta = refine_delta[roi_offset]
                    # # decode refine box
                    # pred_center = roi_center + delta[:3]
                    # pred_size = torch.exp(delta[3:6])

                    delta = refine_delta[roi_offset]
                    refined_box = self.decode_roi_residual(roi.unsqueeze(0), delta.unsqueeze(0))  # (1,6)
                    # pdb.set_trace()

                    if refine_score is None:
                        refined_score = score_mat
                    else:
                        refined_score = refine_score[roi_offset].sigmoid().view(1, 1)

                    refined_box3d = img_metas[i]['box_type_3d'](
                        refined_box,
                        box_dim=6,
                        with_yaw=False,
                        origin=(.5, .5, .5)
                    )

                    new_results.append((refined_box3d, refined_score, label))
                    roi_offset += roi_count

                results = new_results

            # pdb.set_trace()
            seg_masks = self._get_instances(seg_preds[:, 0], idxs[:, 0], v2r, r2scene, scores, gt_idxs, inverse_mapping, img_metas)
            # pdb.set_trace()
            # seg_preds_list, targets_list = [],[]
            
            # for i in range(len(img_metas)):
            #     seg_pred = seg_preds[v2r==i]
            #     target = targets[[v2r==i]]
            #     seg_preds_list.append(seg_pred)
            #     targets_list.append(target)
        else:
            seg_masks = None
        head_time = time.perf_counter() - start_time
        self.last_stage_profile['head_post'] = head_time
        return results, head_time, seg_masks

    def _get_instances(self, cls_preds, idxs, v2r, r2scene, scores, labels, inverse_mapping, img_metas):
        v2scene = r2scene[v2r]
        results = []
        # pdb.set_trace()
        for i in range(len(img_metas)):
            seg_mask, _, _ = self._get_instances_single(
                cls_preds=cls_preds[v2scene == i],
                idxs=idxs[v2scene == i],
                v2r=v2r[v2scene == i],
                scores=scores[i],
                labels=labels[i],
                inverse_mapping=inverse_mapping)
            results.append(seg_mask.squeeze())
        return results
    
    def _get_instances_single(self, cls_preds, idxs, v2r, scores, labels, inverse_mapping):
        if scores.shape[0] == 0:
            return (inverse_mapping.new_zeros((1, len(inverse_mapping)), dtype=torch.bool),
                    inverse_mapping.new_tensor([0], dtype=torch.long),
                    inverse_mapping.new_tensor([0], dtype=torch.float32))
        v2r = v2r - v2r.min()
        assert len(torch.unique(v2r)) == scores.shape[0]
        assert torch.all(torch.unique(v2r) == torch.arange(0, v2r.max() + 1).to(v2r.device))
        # pdb.set_trace()
        cls_preds = cls_preds.sigmoid()
        binary_cls_preds = cls_preds > self.seg_thr
        v2r_one_hot = torch.nn.functional.one_hot(v2r).bool()
        n_rois = v2r_one_hot.shape[1]
        # todo: why convert from float to long here? can it be long or even int32 before this function?
        idxs_expand = idxs.unsqueeze(-1).expand(idxs.shape[0], n_rois).long()
        # todo: can we not convert to ofloat here?
        binary_cls_preds_expand = binary_cls_preds.unsqueeze(-1).expand(binary_cls_preds.shape[0], n_rois)
        cls_preds[cls_preds <= self.seg_thr] = 0
        cls_preds_expand = cls_preds.unsqueeze(-1).expand(cls_preds.shape[0], n_rois)
        idxs_expand[~v2r_one_hot] = inverse_mapping.max() + 1

        # toso: idxs is float. can these tensors be constructed with .new_zeros(..., dtype=bool) ?
        voxels_masks = idxs.new_zeros(inverse_mapping.max() + 2, n_rois, dtype=bool)
        voxels_preds = idxs.new_zeros(inverse_mapping.max() + 2, n_rois)
        voxels_preds = voxels_preds.scatter_(0, idxs_expand, cls_preds_expand)[:-1, :]
        # todo: is it ok that binary_cls_preds_expand is float?
        voxels_masks = voxels_masks.scatter_(0, idxs_expand, binary_cls_preds_expand)[:-1, :]
        scores = scores * voxels_preds.sum(axis=0) / voxels_masks.sum(axis=0)
        points_masks = voxels_masks[inverse_mapping].T.bool()
        return points_masks, labels, scores
    
class TR3DAssigner:
    def __init__(self, top_pts_threshold, top_pts_threshold_det, label2level):
        # top_pts_threshold: per box
        # label2level: list of len n_classes
        #     scannet: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        #     sunrgbd: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        #       s3dis: [1, 0, 1, 1, 0]
        self.top_pts_threshold = top_pts_threshold
        self.top_pts_threshold_det = top_pts_threshold_det
        self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta):
        # pdb.set_trace()
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        if n_boxes>1:
            top_pts_threshold = self.top_pts_threshold_det
        else:
            top_pts_threshold = self.top_pts_threshold
        # pdb.set_trace()
        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3.0: only closest object to point
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)
        # pdb.set_trace()
        return min_inds
    
def get_face_distances(points, boxes):
    # points: of shape (..., 3)
    # boxes: of shape (..., 7)
    # -> of shape (..., 6): dx_min, dx_max, dy_min, dy_max, dz_min, dz_max
    shift = torch.stack((
        points[..., 0] - boxes[..., 0],
        points[..., 1] - boxes[..., 1],
        points[..., 2] - boxes[..., 2]), dim=-1).permute(1, 0, 2)
    shift = rotation_3d_in_axis(shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
    centers = boxes[..., :3] + shift
    dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
    dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
    dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
    dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
    dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
    dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
    return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)


def axis_aligned_iou_3d(boxes1, boxes2, eps=1e-6):
    """
    boxes1: (N, 6) [cx,cy,cz,w,h,l]
    boxes2: (M, 6)
    return: IoU (N, M)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    b1_min = boxes1[:, :3] - boxes1[:, 3:6] / 2
    b1_max = boxes1[:, :3] + boxes1[:, 3:6] / 2
    b2_min = boxes2[:, :3] - boxes2[:, 3:6] / 2
    b2_max = boxes2[:, :3] + boxes2[:, 3:6] / 2

    # (N, M, 3)
    inter_min = torch.maximum(b1_min[:, None, :], b2_min[None, :, :])
    inter_max = torch.minimum(b1_max[:, None, :], b2_max[None, :, :])
    inter = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter[:, :, 0] * inter[:, :, 1] * inter[:, :, 2]

    v1 = (b1_max - b1_min).clamp(min=0)
    v2 = (b2_max - b2_min).clamp(min=0)
    vol1 = v1[:, 0] * v1[:, 1] * v1[:, 2]
    vol2 = v2[:, 0] * v2[:, 1] * v2[:, 2]

    union = vol1[:, None] + vol2[None, :] - inter_vol
    return inter_vol / (union + eps)


def iou_guided_quality(iou):
    """
    iou: Tensor arbitrary shape in [0,1]
    returns q same shape, per your formula:
      1 if iou>0.75
      0 if iou<0.25
      2*iou-0.5 otherwise
    """
    q = 2 * iou - 0.5
    q = torch.where(iou > 0.75, torch.ones_like(q), q)
    q = torch.where(iou < 0.25, torch.zeros_like(q), q)
    return q.clamp(0, 1)
