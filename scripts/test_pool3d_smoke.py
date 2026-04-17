import importlib
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pool3d smoke test")
def test_pool3d_smoke():
    m = importlib.import_module('models.roiaware_pool3d_utils')
    cls = getattr(m, 'RoIAwarePool3d')
    device = torch.device('cuda')
    N = 2
    num_pts = 128
    C = 4
    rois = torch.zeros((N, 7), device=device, dtype=torch.float32)
    rois[:, 3:6] = 1.0  # dx,dy,dz
    pts = torch.rand((num_pts, 3), device=device, dtype=torch.float32)
    pts_feature = torch.rand((num_pts, C), device=device, dtype=torch.float32)
    pool = cls(out_size=3)
    out = pool(rois, pts, pts_feature)
    assert out.shape == (N, 3, 3, 3, C)
