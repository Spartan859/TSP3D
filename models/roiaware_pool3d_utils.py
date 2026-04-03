import importlib
import os
import warnings

import torch
import torch.nn as nn
from torch.autograd import Function


def _select_pool3d_module():
    override = os.getenv("POOL3D_VARIANT", "").strip().lower()
    if override in {"4090", "rtx4090", "pool3d_4090"}:
        module_path = ".pool3d_4090.roiaware_pool3d_cuda"
    elif override in {"h20", "pool3d", "pool3d_h20"}:
        module_path = ".pool3d.roiaware_pool3d_cuda"
    else:
        module_path = ".pool3d.roiaware_pool3d_cuda"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
            if "4090" in device_name:
                module_path = ".pool3d_4090.roiaware_pool3d_cuda"
            elif "h20" in device_name:
                module_path = ".pool3d.roiaware_pool3d_cuda"
            else:
                warnings.warn(
                    "Unknown GPU type for pool3d selection; defaulting to pool3d.",
                    RuntimeWarning,
                )

    return importlib.import_module(module_path, package=__package__)


roiaware_pool3d_cuda = _select_pool3d_module()


class RoIAwarePool3dFunction(Function):
    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel=128, pool_method='avg'):
        """
        rois: (N, 7) [cx, cy, cz, dx, dy, dz, heading]
        pts: (npoints, 3)
        pts_feature: (npoints, C)
        out_size: int or tuple/list of 3 ints
        """
        assert rois.is_cuda and pts.is_cuda and pts_feature.is_cuda
        assert rois.is_contiguous() and pts.is_contiguous() and pts_feature.is_contiguous()

        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[1]

        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = rois.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int32)
        pts_idx_of_voxels = rois.new_zeros(
            (num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int32)

        pool_method_map = {'max': 0, 'avg': 1}
        pool_method_id = pool_method_map[pool_method]

        roiaware_pool3d_cuda.forward(
            rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method_id
        )

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, pts_feature.shape[0], pool_method_id)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        pts_idx_of_voxels, argmax, num_pts, pool_method_id = ctx.roiaware_pool3d_for_backward
        grad_in = grad_out.new_zeros((num_pts, grad_out.shape[-1]))
        roiaware_pool3d_cuda.backward(
            pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method_id
        )
        return None, None, grad_in, None, None, None


class RoIAwarePool3d(nn.Module):
    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='avg'):
        return RoIAwarePool3dFunction.apply(
            rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method
        )