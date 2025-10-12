import numpy as np
from typing import List

import MinkowskiEngine as ME

import torch,time
from mmcv.ops import nms3d, nms3d_normal
from torch import nn

from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
from .axis_aligned_iou_loss import AxisAlignedIoULoss2
from mmdet.models.losses import FocalLoss, DiceLoss
from .trans_modules import (BiEncoder, BiEncoderLayer, BiEncoderLayerSwin, BiEncoderSwin, PositionEmbeddingLearned)
from .mink_unet import MinkUNet14B

import pdb
import logging
import torch.distributed as dist

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
                 pts_prune_threshold=(1200,3600),
                 volume_threshold=27,
                 r=(13,13),
                 assign_type='volume',
                 prune_threshold=(0.3,0.7),
                 com_threshold = 0.15,
                 seg_thr = 0.3,
                 train_cfg=None,
                 test_cfg=dict(nms_pre=1, iou_thr=.5, score_thr=.01),
                 keep_loss_weight = 1.0,
                 bbox_loss_weight = 1.0,
                 seg_loss_weight = 2.,
                 seg_loss_dice_weight = .1,
                 M_q_loss_weight = 2.,
                 M_q_loss_dice_weight = .1,
                 cross_loss_weight = 0.,
                 use_F3_CA = False,
                 swin_drop_path = 0.0,
                 window_size = 5,
                 quant_size = 4,
                 swin_layer_num = 2,
                 use_Swin = False,
                 use_seg = False,
                 use_Mq = -1,
                 use_external_attn_bi_layer0=False,
                 ):
        super(TSPHead, self).__init__()
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
        self.M_q_loss_weight = M_q_loss_weight
        self.M_q_loss_dice_weight = M_q_loss_dice_weight
        self.cross_loss_weight = cross_loss_weight
        self.use_F3_CA = use_F3_CA
        self.swin_drop_path = swin_drop_path
        self.window_size = window_size
        self.quant_size = quant_size
        self.swin_layer_num = swin_layer_num
        self.use_Swin = use_Swin
        self.use_seg = use_seg
        self.use_Mq = use_Mq
        self.use_external_attn_bi_layer0 = use_external_attn_bi_layer0
        self.assigner = TR3DAssigner(top_pts_threshold=24, top_pts_threshold_det=8, label2level=[0])
        self.bbox_loss = AxisAlignedIoULoss2(mode='diou', reduction='none')
        self.cls_loss = FocalLoss(reduction='none')
        self.com_loss = FocalLoss(reduction='none')
        self.keep_loss = FocalLoss(reduction='mean', use_sigmoid=True)
        self.seg_loss = FocalLoss(reduction='mean', use_sigmoid=True)
        self.seg_loss_dice = DiceLoss(reduction='mean', use_sigmoid=True)
        self.M_q_loss = FocalLoss(reduction='mean', use_sigmoid=True)
        self.M_q_loss_dice = DiceLoss(reduction='mean', use_sigmoid=True)
        self.cross_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_samples = (1600,400)
        self.num_samples_com = 1600
        self.com_threshold = com_threshold
        self.random_prune_threshold = (1000,3000)
        self.seg_thr = seg_thr
        
        self.padding = 0.08
        self.min_pts_threshold = 16
        self.max_seg_bbox = 36
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)


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
        # self.seg_conv = ME.MinkowskiConvolution(
        #     out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
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
            use_external_attn=self.use_external_attn_bi_layer0
        )
        if self.use_Swin:
            bi_layer0_swin = BiEncoderLayerSwin(
                128, dropout=0.1, activation="relu",
                n_heads=8, dim_feedforward=128,
                self_attend_lang=True, self_attend_vis=True,
                use_butd_enc_attn=False,
                use_F3_CA=self.use_F3_CA,
                swin_drop_path=self.swin_drop_path,
                window_size=self.window_size,
                quant_size=self.quant_size,
                swin_layer_num=self.swin_layer_num,
            )
        bi_layer1 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        bi_layer2 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        # self.keep_trans = nn.ModuleList([BiEncoder(bi_layer0, 2), BiEncoder(bi_layer1, 2)])
        if self.use_Swin:
            self.keep_trans = nn.ModuleList([bi_layer0_swin, BiEncoder(bi_layer1, 2)])
        else:
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

        if self.use_seg:
            pdb.set_trace()
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

        if self.use_Mq>=0:
            # For M_q branch: learnable scale and bias to convert similarity to logit
            self.mq_scale = nn.Parameter(torch.tensor(10.0))
            self.mq_bias = nn.Parameter(torch.tensor(0.0))
        
        
        # self.maxpool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)


    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))
        # nn.init.normal_(self.seg_conv.kernel, std=.01)
        # nn.init.constant_(self.seg_conv.bias, bias_init_with_prob(.01))
        
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


    def _forward_single(self, x: ME.SparseTensor):
        reg_final = self.bbox_conv(x).features
        # print("Max value before exp:", reg_final[:, 3:6].max().item())
        clamped_reg_final = torch.clamp(reg_final[:, 3:6], max=10.0) # 将输入的最大值限制为10
        reg_distance = torch.exp(clamped_reg_final)
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        scores: ME.SparseTensor = self.cls_conv(x)
        # seg_scores = self.seg_conv(x)
        cls_pred = scores.features

        bbox_preds, cls_preds, points, center_coords, center_bbox_pred = [], [], [], [], []
        for permutation in x.decomposition_permutations:
            # 提取当前场景的分类分数
            scene_cls_pred = cls_pred[permutation]
            
            if scene_cls_pred.numel()>0:
                # 1. 找到当前场景中分类分数最高的体素的局部索引
                best_local_index = torch.argmax(scene_cls_pred)
                best_coord = x.coordinates[permutation][best_local_index]
                best_bbox_pred = bbox_pred[permutation][best_local_index]
                center_coords.append(best_coord)
                center_bbox_pred.append(best_bbox_pred)
            else:
                center_coords.append(torch.zeros(4, device=x.device))
            
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:]* self.voxel_size)
            # pdb.set_trace()
        return bbox_preds, cls_preds, points, center_coords, center_bbox_pred#, seg_scores


    def _get_best_feats_F2(self, center_coords: List[torch.Tensor], saved_xs: List[ME.SparseTensor], kernel_size: int = 3) -> List[torch.Tensor]:
        """
        For each center_coord from F2, find its influential features in F1 and average them.
        x_F1 is the result of a transposed convolution on x_F2 with stride 2.
        """
        x_F2 = saved_xs[-2]
        x_F1 = saved_xs[-1]
        best_feats_F2 = []

        # Decompose the sparse tensor to process each batch item individually
        x1_coords_decomposed: torch.Tensor = x_F1.decomposed_coordinates
        x1_feats_decomposed: torch.Tensor = x_F1.decomposed_features

        for i, center_coord in enumerate(center_coords):
            # center_coord is a single coordinate from x_F2, e.g., [batch_idx, x, y, z]
            # Note: The batch_idx in center_coord is the absolute batch index.
            
            # Get the corresponding coordinates and features for the current batch item from x_F1
            x1_coords = x1_coords_decomposed[i]
            x1_feats = x1_feats_decomposed[i]

            if x1_coords.shape[0] == 0:
                # If there are no points in the x_F1 for this batch, append a zero tensor
                best_feats_F2.append(torch.zeros(x_F1.features.shape[1], device=x_F1.device))
                continue

            # Calculate the center of the search region in x_F1's coordinate space
            # Stride is 2, so we multiply spatial coordinates by 2
            center_in_x1_space = center_coord[1:]
            
            # pdb.set_trace()
            
            stride = torch.tensor(x_F1.tensor_stride, device=x_F1.device)

            # Define the 3x3x3 neighborhood bounds
            lower_bound = center_in_x1_space - (kernel_size // 2) * stride
            upper_bound = center_in_x1_space + (kernel_size // 2) * stride

            # Create a boolean mask to find all coordinates within the neighborhood
            mask = (x1_coords[:, 0] >= lower_bound[0]) & (x1_coords[:, 0] <= upper_bound[0]) & \
                   (x1_coords[:, 1] >= lower_bound[1]) & (x1_coords[:, 1] <= upper_bound[1]) & \
                   (x1_coords[:, 2] >= lower_bound[2]) & (x1_coords[:, 2] <= upper_bound[2])
            # pdb.set_trace()
            # Get the features of the points within the neighborhood
            neighbor_feats = x1_feats[mask]

            if neighbor_feats.shape[0] > 0:
                # If neighbors are found, compute their average feature
                avg_feat = neighbor_feats.mean(dim=0)
                best_feats_F2.append(avg_feat)
            else:
                # If no neighbors are found, append a zero tensor as a placeholder
                best_feats_F2.append(torch.zeros(x_F1.features.shape[1], device=x_F1.device))
        
        return best_feats_F2
            
            
    def _calculate_M_q(self, query_vectors: List[torch.Tensor], supervoxels_tensor: ME.SparseTensor):
        """
        计算查询向量和超体素集合之间的相似度图 M_q。

        Args:
            query_vectors (list[Tensor]): 每个场景的查询向量 Q_box 列表。
            supervoxels_tensor (ME.SparseTensor): 包含整个批次超体素的稀疏张量。

        Returns:
            ME.SparseTensor: 相似度图 M_q。其坐标与supervoxels_tensor相同，
                            特征为每个超体素与对应Q_box的相似度分数。
        """
        if supervoxels_tensor is None:
            return None
        # 1. 从稀疏张量中提取所有超体素的特征
        # supervoxel_features 的形状为 (N_total_supervoxels, 128)
        supervoxel_features = supervoxels_tensor.F

        # 2. 初始化一个张量，用于存放最终计算出的所有相似度分数
        # 这个张量的大小将和supervoxel_features的行数相同
        similarity_scores = torch.zeros(
            supervoxel_features.shape[0], 1, device=supervoxels_tensor.device
        )

        # 3. 遍历批次中的每一个场景
        # enumerate(supervoxels_tensor.decomposition_permutations) 会同时提供
        # 场景索引(i, 从0到7)和该场景对应的索引掩码(permutation)
        for i, permutation in enumerate(supervoxels_tensor.decomposition_permutations):
            
            # 3.1. 获取当前场景的查询向量 Q_box
            # query_vectors[i] 的形状是 (128,)
            q_box = query_vectors[i]

            # 3.2. 使用permutation索引，提取出只属于当前场景的超体素特征
            # scene_supervoxel_features 的形状为 (N_scene_supervoxels, 128)
            scene_supervoxel_features = supervoxel_features[permutation]

            # 3.3. 计算相似度（点积）
            # 这是MCLN论文中 M_q = Q_box × V_s 的实现
            # 我们将 q_box 变形为 (128, 1) 以进行矩阵乘法
            # scene_similarity 的形状为 (N_scene_supervoxels, 1)
            # 计算余弦相似度
            q_box_norm = q_box / (q_box.norm(p=2) + 1e-8)
            scene_supervoxel_features_norm = scene_supervoxel_features / (scene_supervoxel_features.norm(p=2, dim=1, keepdim=True) + 1e-8)
            scene_similarity = (scene_supervoxel_features_norm @ q_box_norm.unsqueeze(1))

            # 3.4. 将计算出的当前场景的相似度分数，放回总的similarity_scores张量的正确位置
            similarity_scores[permutation] = scene_similarity

        # 4. 创建最终的 M_q 稀疏张量
        # 它使用与输入超体素张量完全相同的坐标管理器和坐标图键，
        # 只是将其特征替换为我们刚刚计算出的相似度分数。
        # pdb.set_trace()
        M_q = ME.SparseTensor(
            features=similarity_scores,
            coordinate_map_key=supervoxels_tensor.coordinate_map_key,
            coordinate_manager=supervoxels_tensor.coordinate_manager,
        )

        return M_q


    def _extract_xF2_subset(self, center_bbox_scene_list: List[torch.Tensor], saved_xs: List[ME.SparseTensor], kernel_size: int = 3):
        """根据预测的bbox在F1尺度上的体素范围反推其在F2尺度上的相关体素，提取x_F2的子集。
        步骤:
          1. 把center_bbox_scene_list转成F2的索引尺度 (除以 voxel_size)。
          2. 考虑反卷积 kernel 的 3x3x3 覆盖(向外扩一圈)，得到 F2 尺度的中心坐标和尺寸bbox。
          3. 在x_F2的坐标中筛选所有落入F2尺度中心bbox的体素坐标，形成子集SparseTensor。
        Args:
            center_bbox_scene_list: 每个场景的预测中心 bbox (米制) 列表, 每个元素为 (B,6) 的张量 (cx,cy,cz,w,h,l)
            saved_xs: 已保存的多尺度 SparseTensor 列表, 末尾 -2 为 x_F2, -1 为 x_F1
            kernel_size: 反卷积 kernel 边长(默认3)
        Returns:
            ME.SparseTensor: 只包含贡献F1 中心bbox区域的 x_F2 子集
        """
        if len(saved_xs) < 2:
            return None
        x_F2 = saved_xs[-2]
        x_F1 = saved_xs[-1]
        coords_F2 = x_F2.coordinates  # (N2,4) [b,x,y,z]
        stride_F1 = x_F1.tensor_stride[0] if isinstance(x_F1.tensor_stride, (list, tuple)) else x_F1.tensor_stride
        # F2 更粗, 通过 transposed conv(stride s = stride_F2/stride_F1) 上采到 F1
        expand = stride_F1 * (kernel_size - 1) / 2.0  # 3 -> 1
        device = coords_F2.device
        keep_mask = torch.zeros(coords_F2.shape[0], dtype=torch.bool, device=device)
        # 为了高效: 先按 batch 分组索引
        # 构建 batch -> indices 映射
        batch_indices = {}
        for idx in range(coords_F2.shape[0]):
            b = int(coords_F2[idx,0].item())
            batch_indices.setdefault(b, []).append(idx)
            
        for scene_id, center_bbox_scene in enumerate(center_bbox_scene_list):
            # 1. 转成 F2 索引尺度
            centers_F2 = center_bbox_scene[0:3] / self.voxel_size  # (Ni,3)
            sizes_F2 = center_bbox_scene[3:6] / self.voxel_size    # (Ni,3)
            # 2. 考虑反卷积 kernel 的 3x3x3 覆盖 (向外扩一圈)
            # 直接在 F2 尺度尺寸上加 2*expand
            sizes_F2_expanded = sizes_F2 + 2 * expand
            half_sizes = sizes_F2_expanded / 2.0
            # 4. 获取该 scene 在 F2 中的所有坐标
            scene_indices = batch_indices.get(scene_id, [])
            if not scene_indices:
                continue
            scene_indices_tensor = torch.tensor(scene_indices, device=device, dtype=torch.long)
            scene_coords = coords_F2[scene_indices_tensor][:,1:].float()  # (M,3)
            # 5. 筛选落入 bbox 范围内的坐标
            for i in range(3):
                cond = (scene_coords[:,i] >= (centers_F2[i] - half_sizes[i])) & \
                       (scene_coords[:,i] <= (centers_F2[i] + half_sizes[i]))
                scene_coords = scene_coords[cond]
                scene_indices_tensor = scene_indices_tensor[cond]
                if scene_coords.shape[0] == 0:
                    break
            keep_mask[scene_indices_tensor] = True
            # pdb.set_trace()
        if keep_mask.sum() == 0:
            return None
        x_F2_subset = ME.SparseTensor(
            features=x_F2.features[keep_mask],
            coordinates=x_F2.coordinates[keep_mask],
            coordinate_manager=x_F2.coordinate_manager,
            tensor_stride=x_F2.tensor_stride,
            device=x_F2.device
        )
        return x_F2_subset

    def _generate_Mq_seg(self, M_q: ME.SparseTensor, seg_feats: ME.SparseTensor, radius: int = 5):
        """
        根据 M_q 的分数，为 seg_feats 的每个坐标点生成一个新的分数张量 M_q_seg。
        此版本使用 softmax 进行加权融合，以保证梯度可以反向传播。
        [FIXED] 增加了对无邻居点的处理，防止 softmax 出现 NaN。

        规则:
        对于 seg_feats 中的每个点，它会查看其周围 (2r+1)^3 区域内的所有 M_q 点。
        然后根据这些 M_q 点的分数计算一个 softmax 权重，并用这些权重对分数进行加权求和，
        得到该 seg_feats 点的新分数。

        Args:
            M_q (ME.SparseTensor): 源分数张量。其特征梯度可以被计算。
            seg_feats (ME.SparseTensor): 目标坐标张量。
            radius (int): 扩散的半径 (包含边界)。

        Returns:
            ME.SparseTensor: 一个与 seg_feats 形状完全相同的新稀疏张量 M_q_seg，且可微分。
        """
        # 1. 获取输入张量的坐标和 M_q 的特征
        coords_seg = seg_feats.C
        coords_Mq = M_q.C
        feats_Mq = M_q.F
        device = seg_feats.device

        # 2. 初始化一个新的特征张量，与 seg_feats 的点数相同。
        new_seg_feats = torch.zeros((len(coords_seg), 1), device=device, dtype=torch.float32)

        # 3. 按场景（batch item）进行处理
        for scene_id, seg_perm in enumerate(seg_feats.decomposition_permutations):
            mq_scene_mask = (coords_Mq[:, 0] == scene_id)
            if not torch.any(mq_scene_mask):
                continue

            coords_seg_scene = coords_seg[seg_perm][:, 1:]
            coords_Mq_scene = coords_Mq[mq_scene_mask][:, 1:]
            feats_Mq_scene = feats_Mq[mq_scene_mask]

            # 4. 计算坐标差值并找到在半径内的点
            delta = coords_seg_scene.unsqueeze(1) - coords_Mq_scene.unsqueeze(0)
            is_within_box = (delta.abs() <= radius).all(dim=2)

            # 5. [FIX] 检查哪些 seg_feats 点至少有一个 M_q 邻居
            has_neighbors_mask = is_within_box.any(dim=1)
            
            # 初始化当前场景的分数为0
            weighted_scores = torch.zeros(len(coords_seg_scene), device=device)

            # 只对有邻居的点进行 softmax 计算
            if has_neighbors_mask.any():
                # 过滤出有邻居的点进行后续计算
                relevant_seg_points_mask = has_neighbors_mask
                
                # 广播 M_q 的分数，并将不在范围内的分数设为-inf以便 softmax 正确处理
                broadcasted_scores = feats_Mq_scene.T.expand(len(coords_seg_scene), -1)
                
                # 只考虑有邻居的行
                scores_to_consider = torch.where(
                    is_within_box[relevant_seg_points_mask],
                    broadcasted_scores[relevant_seg_points_mask],
                    -torch.inf
                )

                # 6. 计算 softmax 权重。-inf 的位置权重会变为 0。
                softmax_weights = torch.softmax(scores_to_consider, dim=1)

                # 7. 使用 softmax 权重对原始分数进行加权求和。
                calculated_scores = torch.sum(
                    broadcasted_scores[relevant_seg_points_mask] * softmax_weights, 
                    dim=1
                )
                
                # 将计算出的分数放回正确的位置
                weighted_scores[relevant_seg_points_mask] = calculated_scores

            # 8. 将计算出的场景分数放回全局特征张量中
            new_seg_feats[seg_perm] = weighted_scores.unsqueeze(1)

        # 9. 创建最终的稀疏张量 M_q_seg
        M_q_seg = ME.SparseTensor(
            features=new_seg_feats,
            coordinate_map_key=seg_feats.coordinate_map_key,
            coordinate_manager=seg_feats.coordinate_manager
        )

        return M_q_seg
    
    def _get_M_q_F2(self, center_coord, center_bbox_pred, saved_xs, device):
        best_feats_F2 = self._get_best_feats_F2(center_coord, saved_xs)
        center_point = [cc[1:] * self.voxel_size for cc in center_coord]  # 提取空间坐标并转为米制
        # 转tensor
        center_point = torch.stack(center_point) if len(center_point) > 0 else torch.empty(0, 3, device=device)
        center_bbox_pred = torch.stack(center_bbox_pred) if len(center_bbox_pred) > 0 else torch.empty(0, 6, device=device)
        # 转换成米制真实框
        center_bbox = self._bbox_pred_to_bbox(center_point, center_bbox_pred)  # center_bbox_pred 是 (B,6) 的偏移+尺度, center_bbox 是 (B,6) 的米制真实框 (cx,cy,cz,w,h,l)
        # 新增: 提取 x_F2 子集 (可能为 None)
        # pdb.set_trace()
        x_F2_subset = self._extract_xF2_subset(center_bbox, saved_xs, kernel_size=3)
        # bbox_F2 = self._get_bbox_F2(bbox_pred, saved_xs)
        # pdb.set_trace()
        M_q = self._calculate_M_q(best_feats_F2, x_F2_subset)
        return M_q
    
    def _get_M_q_F1(self, center_coord, center_bbox_pred, saved_xs, cls_pred,device):
        center_point = [cc[1:] * self.voxel_size for cc in center_coord]  # 提取空间坐标并转为米制
        # 转tensor
        center_point = torch.stack(center_point) if len(center_point) > 0 else torch.empty(0, 3, device=device)
        center_bbox_pred = torch.stack(center_bbox_pred) if len(center_bbox_pred) > 0 else torch.empty(0, 6, device=device)
        # 转换成米制真实框
        center_bbox = self._bbox_pred_to_bbox(center_point, center_bbox_pred)  # (B,6): (cx,cy,cz,w,h,l)

        # 从F1尺度的特征图中提取落在每个scene对应center_bbox内的点
        x_F1: ME.SparseTensor = saved_xs[-1]
        coords_F1 = x_F1.coordinates  # (N, 4): [batch, x, y, z]
        device = x_F1.device

        selected_coords = []
        selected_feats = []

        # 按scene处理：使用与F1相同的分解索引
        for scene_id, perm in enumerate(x_F1.decomposition_permutations):
            if len(perm) == 0:
                continue
            # 防守：若center_bbox数量少于scene数量，跳过越界scene
            if center_bbox.shape[0] <= scene_id:
                continue

            # 该scene的F1点坐标(米制)
            scene_coords_idx = coords_F1[perm]                   # int坐标(保持原dtype用于构建SparseTensor)
            scene_points_m = scene_coords_idx[:, 1:].float() * self.voxel_size  # 转为米制用于几何判断

            # 当前scene的bbox参数
            cx, cy, cz, w, h, l = center_bbox[scene_id]
            half = torch.tensor([w, h, l], device=device, dtype=scene_points_m.dtype) / 2.0
            center = torch.tensor([cx, cy, cz], device=device, dtype=scene_points_m.dtype)

            lower = center - half
            upper = center + half

            # 选出落在bbox范围内的点
            mask = (scene_points_m[:, 0] >= lower[0]) & (scene_points_m[:, 0] <= upper[0]) & \
                   (scene_points_m[:, 1] >= lower[1]) & (scene_points_m[:, 1] <= upper[1]) & \
                   (scene_points_m[:, 2] >= lower[2]) & (scene_points_m[:, 2] <= upper[2])

            if mask.any():
                # 从cls_pred（按scene分割的logits或分数）中取出对应点的分数作为新特征
                # cls_pred[scene_id] 形状: (Ni, 1)（n_classes=1时）
                scene_scores = cls_pred[scene_id]
                selected_coords.append(scene_coords_idx[mask])
                selected_feats.append(scene_scores[mask])

        if len(selected_coords) == 0:
            # 返回一个空的稀疏张量（坐标管理与F1一致），后续逻辑可正常跳过
            M_q = ME.SparseTensor(
                features=coords_F1.new_zeros((0, 1), dtype=torch.float32),
                coordinates=coords_F1.new_zeros((0, 4)).float(),  # dtype需为float
                coordinate_manager=x_F1.coordinate_manager,
                tensor_stride=x_F1.tensor_stride,
                device=x_F1.device
            )
            return M_q

        # 拼接被选中的F1子集坐标与对应的分数作为特征
        selected_coords = torch.cat(selected_coords, dim=0).float()  # ME 需要 float 坐标
        selected_feats = torch.cat(selected_feats, dim=0).to(torch.float32)

        # 构建最终的 M_q 稀疏张量（只包含bbox范围内的点，特征为分类分数）
        M_q = ME.SparseTensor(
            features=selected_feats,
            coordinates=selected_coords,
            coordinate_manager=x_F1.coordinate_manager,
            tensor_stride=x_F1.tensor_stride,
            device=x_F1.device
        )
        return M_q

    def _avg_feats_in_stride(self, sampled_coords: torch.Tensor, coords_x: ME.SparseTensor, tensor_stride: int | list[int] | torch.Tensor):
        """
        对于sampled_coords中的每个点，在其±tensor_stride的范围内查找coords_x中的点（同一batch），
        并对找到的所有点的feat做平均，返回coords_vis。
        若坐标的batch为-1，则特征直接设为全0。
        sampled_coords: [B, N, 4] int/float
        coords_x: ME.SparseTensor，features: [M, C]，coordinates: [M, 4]
        tensor_stride: int 或 [3]，体素步长
        返回: coords_vis [B, N, C]
        """
        B, N, D = sampled_coords.shape
        M, C = coords_x.F.shape
        coords_x_coords = coords_x.C  # [M, 4]
        coords_x_feats = coords_x.F  # [M, C]
        if isinstance(tensor_stride, (list, tuple, torch.Tensor)):
            stride = torch.tensor(tensor_stride, device=coords_x_coords.device)
        else:
            stride = torch.full((3,), tensor_stride, device=coords_x_coords.device)
        coords_vis = []
        for b in range(B):
            batch_mask = (coords_x_coords[:, 0] == b)
            coords_x_b = coords_x_coords[batch_mask, 1:4]  # [Mb, 3]
            feats_x_b = coords_x_feats[batch_mask]          # [Mb, C]
            coords_b = sampled_coords[b]  # [N, 4]
            vis_b = []
            for n in range(N):
                if coords_b[n, 0] == -1:
                    vis_b.append(torch.zeros(C, device=coords_x.F.device, dtype=coords_x.F.dtype))
                    continue
                q = coords_b[n, 1:4]  # [3]
                # 查找在q±stride范围内的所有点
                lower = q - stride
                upper = q + stride
                mask = ((coords_x_b >= lower) & (coords_x_b <= upper)).all(dim=1)
                if mask.any():
                    feats = feats_x_b[mask]
                    vis_b.append(feats.mean(dim=0))
                else:
                    vis_b.append(torch.zeros(C, device=coords_x.F.device, dtype=coords_x.F.dtype))
            vis_b = torch.stack(vis_b, dim=0)  # [N, C]
            coords_vis.append(vis_b)
        coords_vis = torch.stack(coords_vis, dim=0)  # [B, N, C]
        return coords_vis
    
    def forward(self, x_all, coords_x: ME.SparseTensor, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas, pc=None):
        # pdb.set_trace()
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
        saved_xs = []
        x_F3_pos_feats = None
        x_F3_vis_feats = None
        x_F3_padding_mask = None
        for i in range(len(inputs) - 1, -1, -1): # 2,1,0
            if i ==1 :  #  1,0         
                prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas) 

                keep_gt = []
                for permutation in x.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                # pdb.set_trace()
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
                # pdb.set_trace()
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
                    # print(i, len(permutation))
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
                pos_feats = self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous()
                vis_feats = sampled_features.contiguous()
                padding_mask=sampled_coords[:, :,0] == -1
                # coords_vis = self._avg_feats_in_stride(sampled_coords, coords_x, 5)  # [B, N, C]
                
                # pdb.set_trace()
                if x_F3_pos_feats is None:
                    x_F3_pos_feats = pos_feats
                    x_F3_padding_mask = padding_mask
                
                if i < 2 and self.use_Swin:
                    sampled_features, text_feats, x_F3_vis_feats = self.keep_trans[i-1](
                        vis_feats=vis_feats,
                        pos_feats=pos_feats,
                        coords_vis=sampled_coords.float(),
                        sampled_coords=sampled_coords,
                        x_F3_vis_feats=x_F3_vis_feats,
                        x_F3_pos_feats=x_F3_pos_feats,
                        padding_mask=padding_mask,
                        x_F3_padding_mask=x_F3_padding_mask,
                        text_feats=text_feats,
                        text_padding_mask=text_attention_mask)
                else:
                    sampled_features, text_feats = self.keep_trans[i-1](
                        vis_feats=vis_feats,
                        pos_feats=pos_feats,
                        padding_mask=padding_mask,
                        text_feats=text_feats,
                        text_padding_mask=text_attention_mask)
                    x_F3_vis_feats = sampled_features
                # pdb.set_trace()
                
                
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                
                x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords, 
                                    coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)
                # pdb.set_trace()
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
            # if not dist.is_initialized() or dist.get_rank() == 0:
            #     pdb.set_trace()                
            x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                out = self.__getattr__(f'out_block_{i}')(x)
            saved_xs.append(x)
        out = self.fuse(out, text_feats[:, 0])
        bbox_pred, cls_pred, point, center_coord, center_bbox_pred = self._forward_single(out)
        # pdb.set_trace()
        
        if self.use_Mq==2:
            M_q = self._get_M_q_F2(center_coord, center_bbox_pred, saved_xs, device=x.device)
        elif self.use_Mq==1:
            M_q = self._get_M_q_F1(center_coord, center_bbox_pred, saved_xs, cls_pred, device=x.device)
        elif self.use_Mq==-1:
            M_q = None
        else:
            logging.error('use_Mq can only be 2, 1 or -1')
            pdb.set_trace()
        
        if self.use_seg:
            # pdb.set_trace()
            x = self.upsample_st_2(x) + x_all[1]
            x = self.upsample_st_4(x) + x_all[0]
            seg_feats = self.conv_32_ch(x)
            if M_q is not None and M_q.F.shape[0] > 0:
                M_q_seg = self._generate_Mq_seg(M_q, seg_feats, radius=5)
            else:
                M_q_seg = None
        else:
            seg_feats = None
            M_q_seg = None
        # pdb.set_trace()
            
        return [bbox_pred], [cls_pred], [point], keep_preds[::-1], keep_gts[::-1], bboxes_level, com_pred_training, com_coords_training, \
            seg_feats, M_q_seg


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

            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > self.prune_threshold[layer_id]
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True                 
        if prune_mask.sum() != 0:
            x = self.pruning(x, prune_mask)
        else:
            x = None

        return x


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
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
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

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        # pdb.set_trace()
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
              keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training, gt_points, targets, seg_feats, M_q_seg):
        bbox_losses, cls_losses, pos_masks, com_losses, pos_masks_com, selected_bboxes, selected_scores, selected_labels \
            = [], [], [], [], [], [], [], []

        # segmentation loss
        # pdb.set_trace()
        # seg_losses = 0
        # seg_losses_dice = 0
        # k = 0
        # for permutation in seg_scores.decomposition_permutations:
        #     coordinates, features = ME.utils.batch_sparse_collate(
        #         [(gt_points[k][:, :3] / self.voxel_size, gt_masks[k].unsqueeze(1))],
        #         device=seg_scores.device)  
        #     gt_masks_sparse = ME.SparseTensor(features=features, coordinates=coordinates, device=seg_scores.device)
        #     for _ in range(4):
        #         gt_masks_sparse = self.maxpool(gt_masks_sparse)
        #     # pdb.set_trace()
        #     gt_masks_sparse = gt_masks_sparse.features_at_coordinates(seg_scores.coordinates[permutation].float())
        #     seg_scores_tensor = seg_scores.features[permutation]
        #     gt_mask_num = gt_masks_sparse.sum()
        #     if gt_mask_num != 0:
        #         seg_loss = self.seg_loss(seg_scores_tensor.squeeze(), gt_masks_sparse.long().squeeze(), avg_factor=gt_mask_num)
        #         seg_loss_dice = self.seg_loss_dice(seg_scores_tensor, gt_masks_sparse.long(), avg_factor=gt_mask_num)
        #     else:
        #         # pdb.set_trace()
        #         seg_loss = self.seg_loss(seg_scores_tensor.squeeze(), gt_masks_sparse.long().squeeze(), avg_factor=len(gt_masks_sparse))
        #         seg_loss_dice = self.seg_loss_dice(seg_scores_tensor, gt_masks_sparse.long(), avg_factor=len(gt_masks_sparse))
        #     seg_losses = seg_losses + seg_loss
        #     seg_losses_dice = seg_losses_dice + seg_loss_dice
        #     k = k + 1       
            # pdb.set_trace()
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
                    # pdb.set_trace()
                else:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=len(gt))  
                    k_loss = torch.mean(keep_loss) / 3 + k_loss

            keep_losses = keep_losses + k_loss

        for i in range(len(img_metas)):
        # for i, (coordinates, features) in enumerate(
        #     zip(*seg_feats.decomposed_coordinates_and_features)):
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
            seg_preds, targets, v2r, r2scene, rois, scores, gt_idxs, M_q_seg_preds = self._forward_seg(seg_feats, targets, selected_bboxes,selected_scores,selected_labels, M_q_seg)
            seg_loss, seg_loss_dice, M_q_loss, M_q_loss_dice, cross_loss = self._loss_second(seg_preds, targets, v2r, r2scene, rois, gt_idxs,gt_bboxes, gt_labels, img_metas, M_q_seg_preds)
        # pdb.set_trace()
        loss_dict = dict(
            bbox_loss=self.bbox_loss_weight * torch.mean(torch.cat(bbox_losses)),
            cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
            keep_loss=self.keep_loss_weight * keep_losses / len(img_metas),
            com_loss=torch.sum(torch.cat(com_losses)) / torch.sum(torch.cat(pos_masks_com)),
        )
        if self.use_seg:
            loss_dict.update(dict(
                seg_loss=self.seg_loss_weight * seg_loss,
                seg_loss_dice=self.seg_loss_dice_weight * seg_loss_dice,
            ))
        if self.use_Mq == 1 or self.use_Mq == 2:
            loss_dict.update(dict(
                M_q_loss=self.M_q_loss_weight * M_q_loss,
                M_q_loss_dice=self.M_q_loss_dice_weight * M_q_loss_dice,
                cross_loss=self.cross_loss_weight * cross_loss
            ))
        return loss_dict

    def _loss_second(self, cls_preds, targets, v2r, r2scene, rois, gt_idxs,
                    gt_bboxes, gt_labels, img_metas, M_q_seg_preds=None):
        # pdb.set_trace()
        try:
            v2scene = r2scene[v2r]
            seg_losses = []
            seg_losses_dice = []
            M_q_losses, M_q_losses_dice = [], []
            cross_losses = []
            for i in range(len(img_metas)):
                seg_loss, seg_loss_dice, M_q_loss, M_q_loss_dice, cross_loss = self._loss_second_single(
                    cls_preds=cls_preds[v2scene == i],
                    targets=targets[v2scene == i],
                    v2r=v2r[v2scene == i],
                    rois=rois[i],
                    gt_idxs=gt_idxs[i],
                    gt_bboxes=gt_bboxes[i],
                    gt_labels=gt_labels[i],
                    img_meta=img_metas[i],
                    M_q_seg_pred=M_q_seg_preds[v2scene == i] if M_q_seg_preds is not None else None
                )
                seg_losses.append(seg_loss)
                seg_losses_dice.append(seg_loss_dice)
                M_q_losses.append(M_q_loss)
                M_q_losses_dice.append(M_q_loss_dice)
                cross_losses.append(cross_loss)

            # pdb.set_trace()
            return torch.mean(torch.stack(seg_losses)), torch.mean(torch.stack(seg_losses_dice)), \
                    torch.mean(torch.stack(M_q_losses)), torch.mean(torch.stack(M_q_losses_dice)), \
                    torch.mean(torch.stack(cross_losses))
        except Exception as e:
            print(e)
            pdb.set_trace()
    
    def _loss_second_single(self, cls_preds, targets, v2r, rois, gt_idxs, gt_bboxes, gt_labels, img_meta, M_q_seg_pred=None):
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
        
        # 初始化 M_q 和 cross loss 为零
        M_q_loss = cls_preds.sum() * 0
        M_q_loss_dice = cls_preds.sum() * 0
        cross_loss = cls_preds.sum() * 0

        if M_q_seg_pred is not None:
            # 2. 将 M_q_seg_pred (相似度) 转换为 logits
            m_q_logits = M_q_seg_pred * self.mq_scale + self.mq_bias

            # 3. 计算 M_q_seg_pred 相对于真值的损失
            M_q_loss = self.M_q_loss(m_q_logits, (targets).long())
            M_q_loss_dice = self.M_q_loss_dice(m_q_logits, (targets).long())

            # 4. 计算协同训练的交叉损失 (对称KL散度)
            # 使用 sigmoid 将 logits 转换为概率 p
            p_seg = torch.sigmoid(cls_preds)
            p_mq = torch.sigmoid(m_q_logits)

            # 构建二分类概率分布 [1-p, p]
            p_dist_seg = torch.cat([1 - p_seg, p_seg], dim=-1)
            p_dist_mq = torch.cat([1 - p_mq, p_mq], dim=-1)
            
            # 为保证数值稳定性，在取对数前给概率值增加一个小的 epsilon
            p_dist_seg = torch.clamp(p_dist_seg, 1e-8, 1.0 - 1e-8)
            p_dist_mq = torch.clamp(p_dist_mq, 1e-8, 1.0 - 1e-8)

            # 计算 log-probabilities
            log_p_dist_seg = torch.log(p_dist_seg)
            log_p_dist_mq = torch.log(p_dist_mq)

            # 计算对称KL散度，梯度会流向两个分支
            loss_seg_to_mq = self.cross_loss(log_p_dist_mq, p_dist_seg)
            loss_mq_to_seg = self.cross_loss(log_p_dist_seg, p_dist_mq)
            
            cross_loss = loss_mq_to_seg + loss_seg_to_mq
            
        return seg_loss, seg_loss_dice, M_q_loss, M_q_loss_dice, cross_loss
    
    def forward_train(self, x, coords_x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, \
        gt_points, targets, img_metas,pc=None):
        
        bbox_preds, cls_preds, points, keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training, seg_feats, M_q_seg = \
            self(x, coords_x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc)

        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level,
                          com_pred_training, com_coords_training, gt_points, targets, seg_feats, M_q_seg)

    def _forward_seg(self, x, targets, rois, scores, labels, M_q_seg: ME.SparseTensor=None):
        # rois = [b[0] for b in bbox_list]
        # scores = [b[1] for b in bbox_list]
        # labels = [b[2] for b in bbox_list]
        # levels = [torch.zeros(len(b[0])) for b in bbox_list]
        # pdb.set_trace()
        feats_with_targets = ME.SparseTensor(torch.cat((x.features, targets), axis=1), x.coordinates)
        if M_q_seg is not None :
            M_q_seg_t , _ , _ , _ , _ = self.extract(M_q_seg, rois, scores, labels)
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
        # pdb.set_trace()
        if self.use_seg:
            preds = self.seg_unet(feats).features
        else:
            preds = None
        # pdb.set_trace()
        return preds, targets, feats.coordinates[:, 0].long(), ids, rois, scores, labels, \
            M_q_seg_t.features if M_q_seg is not None else None

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
            
        new_tensors = ME.SparseTensor(
            torch.cat(new_features),
            torch.cat(new_coordinates).float(),
            tensor_stride=tensors.tensor_stride)
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


    def forward_test(self, x_all, coords_x, text_feats, text_attention_mask, targets, inverse_mapping, img_metas, pc=None, gt_bboxes=None):
        inputs = x_all[2:]
        x = inputs[-1]
        bbox_preds, cls_preds, points = [], [], []
        keep_scores = None
        
        x_F3_pos_feats = None
        x_F3_vis_feats = None
        x_F3_padding_mask = None
        for i in range(len(inputs) - 1, -1, -1):
            if i ==1:
                x = self._prune_inference(x, prune_inference,i)
                
                if x != None:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                else:
                    pdb.set_trace()
                    break
            elif i ==0:
                x = self._prune_inference(x, prune_inference,i)
                
                if x != None:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x_ori = x + x_level
                else:
                    pdb.set_trace()
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
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                
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
                pos_feats = self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous()
                vis_feats = sampled_features.contiguous()
                padding_mask = sampled_coords[:, :,0] == -1
                # coords_vis = self._avg_feats_in_stride(sampled_coords, coords_x, 5)
                
                
                if x_F3_pos_feats is None:
                    x_F3_pos_feats = pos_feats
                    x_F3_padding_mask = padding_mask
                
                if i < 2 and self.use_Swin:
                    sampled_features, text_feats, x_F3_vis_feats = self.keep_trans[i-1](
                        vis_feats=vis_feats,
                        pos_feats=pos_feats,
                        coords_vis=sampled_coords.float(),
                        sampled_coords=sampled_coords,
                        x_F3_vis_feats=x_F3_vis_feats,
                        x_F3_pos_feats=x_F3_pos_feats,
                        padding_mask=padding_mask,
                        x_F3_padding_mask=x_F3_padding_mask,
                        text_feats=text_feats,
                        text_padding_mask=text_attention_mask)
                else:
                    sampled_features, text_feats = self.keep_trans[i-1](
                        vis_feats=vis_feats,
                        pos_feats=pos_feats,
                        padding_mask=padding_mask,
                        text_feats=text_feats,
                        text_padding_mask=text_attention_mask)
                    x_F3_vis_feats = sampled_features
                
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
        start_time = time.time()
        out = self.fuse(out, text_feats[:, 0])
        bbox_pred, cls_pred, point, _ , _ = self._forward_single(out)
        
        results = self._get_bboxes([bbox_pred], [cls_pred], [point], img_metas)
        
        if self.use_seg:
            x = self.upsample_st_2(x) + x_all[1]
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
            seg_preds, idxs, v2r, r2scene, rois, scores, gt_idxs, _ = self._forward_seg(seg_feats, src_idxs.unsqueeze(-1), selected_bboxes,selected_scores,selected_labels)
            # seg_preds, targets_new, v2r, r2scene, rois, scores, gt_idxs = self._forward_seg(seg_feats, targets, selected_bboxes,selected_scores,selected_labels)

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
        head_time = time.time() - start_time
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


