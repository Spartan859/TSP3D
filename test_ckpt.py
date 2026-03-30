import torch
from pathlib import Path

ckpt = Path("scripts/experiments/TGEA_convrefine/TGEA_refine_noscore_loss120/scanrefer/2026-03-22_13-02-49/ckpt_epoch_240.pth")
data = torch.load(ckpt, map_location="cpu")

model = data["model"]
keys = list(model.keys())

print("num_model_keys:", len(keys))
print("first_30_model_keys:", keys[:30])

refine_keys = [k for k in keys if "refine" in k][:50]
print("refine_keys_sample:", refine_keys)

head_keys = [k for k in keys if "head" in k][:50]
print("head_keys_sample:", head_keys)