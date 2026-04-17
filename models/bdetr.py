import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
import MinkowskiEngine as ME
from .mink_resnet import TSPBackbone
from .tr3d_neck import TR3DNeck
from .multilevel_head_refine import TSPHead as TSPHead_refine
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes
from mmdet3d.structures import bbox3d2result
import time
import pdb
    
class BeaUTyDETR(nn.Module):
    """
    3D language grounder.
    """

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=128, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True, voxel_size=0.01,
                 use_seg=False, 
                 use_refine=False,
                 use_seg_external_self_attn=False,
                 use_external_attn_bi_layer=(),
                 use_text_guided_external_attn_bi_layer=(),
                 use_film_text_guided_external_attn_bi_layer=(),
                 com_threshold=0.15,
                 num_samples_com=2400,
                 external_attn_coef=4,
                 mink_conv1_stride=2,
                 top_pts_threshold=None,
                 top_pts_threshold_det=None):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd
        self.voxel_size = voxel_size

        # Visual encoder
        self.vision_backbone = TSPBackbone(in_channels=6, conv1_stride=mink_conv1_stride)
        
        # Text encoder
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )       
        
        # self.neck = TR3DNeck()
        self.head = TSPHead_refine(
            voxel_size=self.voxel_size,
            use_seg=use_seg,
            use_seg_external_self_attn=use_seg_external_self_attn,
            use_external_attn_bi_layer=use_external_attn_bi_layer,
            use_text_guided_external_attn_bi_layer=use_text_guided_external_attn_bi_layer,
            use_film_text_guided_external_attn_bi_layer=use_film_text_guided_external_attn_bi_layer,
            com_threshold=com_threshold,
            num_samples_com=num_samples_com,
            external_attn_coef=external_attn_coef,
            top_pts_threshold=top_pts_threshold,
            top_pts_threshold_det=top_pts_threshold_det
        )
        self.target_pool = None
        if mink_conv1_stride > 1:
            self.target_pool = ME.MinkowskiMaxPooling(
                kernel_size=mink_conv1_stride,
                stride=mink_conv1_stride,
                dimension=3)
        
    def collate(self, points, quantization_mode):
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 0:]) for p in points],
            dtype=points[0].dtype,
            device=points[0].device)
        return ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=quantization_mode,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device,
        )
           
    # BRIEF forward.
    def forward(self, inputs, gt_bboxes=None, gt_labels=None, gt_all_bbox_new=None, auxi_bbox=None, gt_masks=None, img_metas=None, epoch=None):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B
        Returns:
            end_points: dict
        """
        # STEP 1. vision and text encoding
        points = inputs['point_clouds']
        start_time = time.time()
        coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p[:, 0:] if p.shape[1] > 3 else p[:, :3]) for p in points],
                device=points[0].device)        
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        # pdb.set_trace()
        points = [torch.cat([p, torch.unsqueeze(mask, 1)], dim=1) for p, mask in zip(points, gt_masks)]
        field = self.collate(points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        x = field.sparse()
        # pdb.set_trace()
        targets = x.features[:, 6:].round().long()
        if self.target_pool is not None:
            targets = ME.SparseTensor(
                features=targets.float(),
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
            targets = self.target_pool(targets).features
        x = ME.SparseTensor(
            x.features[:, :6],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        # pdb.set_trace()
        x = self.vision_backbone(x)
        # pdb.set_trace()
        inverse_mapping = field.inverse_mapping(x[0].coordinate_map_key).long()
        # pdb.set_trace()
        visual_time = time.time() - start_time
        # pdb.set_trace()
        
        # Text encoding

        start_time = time.time()
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt"
        ).to(inputs['point_clouds'].device)
        
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state) 
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        text_time = time.time() - start_time
        
        if not self.training:
            start_time = time.time()
            bbox_list, head_time, seg_masks = self.head.forward_test(x, text_feats, text_attention_mask, targets, inverse_mapping, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            fusion_time = time.time() - start_time
            return bbox_results, seg_masks, {'loss':0.}, 0., [visual_time,text_time,fusion_time-head_time,head_time]
        losses = self.head.forward_train(x,text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, \
            points, targets, img_metas)
        losses.update({'loss':sum(value for key, value in losses.items() if '_loss' in key)})
        return losses
    
    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
