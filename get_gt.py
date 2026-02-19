import torch 
import numpy as np
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes


def get_gt(end_points):
    gt_labels = end_points['sem_cls_label']
    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)
    box_label_mask = end_points['box_label_mask'] 
    gt_all_bbox = end_points['all_bboxes']
    all_bbox_label_mask = end_points['all_bbox_label_mask'] 
    auxi_box = end_points['auxi_box']
    gt_masks = end_points['gt_masks']
    
    gt_bbox_new = [gt_bbox[b, box_label_mask[b].bool()] for b in range(gt_labels.shape[0])]
    gt_bbox_new = [DepthInstance3DBoxes(box,box_dim=6, with_yaw=False, origin=(.5, .5, .5)) for box in gt_bbox_new]
    gt_labels_new = [gt_labels[b, box_label_mask[b].bool()] for b in range(gt_labels.shape[0])]
    gt_labels_new = [torch.zeros_like(tensor) for tensor in gt_labels_new]
    img_meta = [{'box_type_3d': DepthInstance3DBoxes} for _ in range(len(gt_labels_new))]
    
    gt_all_bbox_new = [gt_all_bbox[b, all_bbox_label_mask[b].bool()] for b in range(gt_labels.shape[0])]
    gt_all_bbox_new = [DepthInstance3DBoxes(box,box_dim=6, with_yaw=False, origin=(.5, .5, .5)) for box in gt_all_bbox_new]
    
    auxi_bbox = [DepthInstance3DBoxes(auxi_box[b],box_dim=6, with_yaw=False, origin=(.5, .5, .5)) if auxi_box[b].max()>0 
                 else DepthInstance3DBoxes([],box_dim=6, with_yaw=False, origin=(.5, .5, .5)) for b in range(gt_labels.shape[0]) ]
    gt_masks_new = [gt_masks[b, box_label_mask[b].bool()] for b in range(gt_masks.shape[0])]
    # import pdb;pdb.set_trace()
    gt_masks_new = [torch.clamp(mask.sum(dim=0), min=0.0, max=1.0) for mask in gt_masks_new]
    
    return gt_bbox_new, gt_labels_new, gt_all_bbox_new, auxi_bbox, gt_masks_new, img_meta