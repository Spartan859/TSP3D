# Final Results Summary

Accuracy results are summarized from `scripts/experiments.md`. Values are reported as percentages. When both training-time and standalone evaluation results are available, all accuracy metrics in a row come from the same result set; train and eval metrics are not mixed.

## ScanRefer Accuracy

| Config | Epoch | Acc@0.25 | Acc@0.50 | Mask@0.25 | Mask@0.50 | Source |
|--------|------:|---------:|---------:|----------:|----------:|--------|
| ori | 204 | 55.86 | 46.42 | -- | -- | `scripts/experiments/TF32/ori/scanrefer/2026-04-19_04-49-09` |
| EA0 | 96 | 56.31 | 45.96 | -- | -- | `scripts/experiments/TF32/EA0/scanrefer/2026-04-18_17-45-08` |
| EA0 (20260216) | 204 | 55.91 | 46.62 | -- | -- | `scripts/experiments/20260216/scanrefer/2026-02-16_11-14-34` |
| EA02 | 294 | 56.31 | 46.20 | -- | -- | `scripts/experiments/TF32/EA02_num_sps_com_2400/scanrefer/2026-04-18_17-51-17` |
| EA0+seg+refine (deprecated) | 267 | 56.29 | 48.45 | 58.22 | 51.30 | `scripts/experiments/TGEA_convrefine/TGEA_refine_noscore/scanrefer/2026-03-21_03-14-46` |
| EA0+seg+refine+tps_thres_fix | 378 | 56.15 | 48.70 | 58.35 | 51.36 | `scripts/experiments/SEG_tps_thres32/SEG_tps_thres_seg_single_multi_32/scanrefer/2026-04-28_17-22-12/ckpt_epoch_378.pth` |
| EA0+seg+refine+tps_thres_fix+SEGEA | 300 | 56.22 | 48.90 | 58.16 | 51.11 | `SEG_tps_thres_seg_single_multi_32_SEGEA_resume2` |
| EA02+seg+refine+tps32 unfreeze | 348 | 56.70 | 48.88 | 58.41 | 51.62 | `scripts/experiments/scanrefer_EA02_SEG_freeze/EA02_SEG_unfreeze/scanrefer/2026-07-05_22-00-52/ckpt_epoch_348.pth` |
| EA02+seg+refine+tps32 freeze resume | 357 | 56.11 | 48.23 | 58.01 | 50.44 | `scripts/experiments/scanrefer_EA02_SEG_freeze/EA02_SEG_freeze_resume/scanrefer/2026-07-05_22-00-38` |
| EA02+seg+refine+tps32 freeze + seg self-attn | 318 | 55.96 | 47.98 | 57.75 | 50.03 | `scripts/experiments/scanrefer_EA02_SEG_freeze/EA02_SEG_freeze_seg_self_attn/scanrefer/2026-07-04_21-16-50` |

Best ScanRefer Acc@0.50 in this summary: `EA0+seg+refine+tps_thres_fix+SEGEA`, 48.90. Best Mask@0.50: `EA02+seg+refine+tps32 unfreeze`, 51.62.

## NR3D Accuracy

| Config | Epoch | Acc@0.25 | Acc@0.50 | Mask@0.25 | Mask@0.50 | Source |
|--------|------:|---------:|---------:|----------:|----------:|--------|
| EA0 | 201 | 49.86 | 38.17 | -- | -- | `scripts/experiments/nr3d/EA0/nr3d/2026-05-18_01-11-03/ckpt_epoch_201.pth` |
| EA02 | 141 | 48.93 | 37.93 | -- | -- | `scripts/experiments/nr3d/EA02/nr3d/2026-05-18_01-13-49/ckpt_epoch_141.pth` |
| EA0+seg | 216 | 50.05 | 39.56 | 51.87 | 42.78 | `EA0_SEG` |
| EA0+seg (freeze) | 249 | 50.33 | 39.84 | 52.64 | 43.51 | `EA0_SEG_freeze` |
| EA02+seg+refine+tps32 unfreeze | 207 | 48.84 | 38.97 | 50.56 | 42.56 | `scripts/experiments/nr3d/EA02_SEG_unfreeze_tps32/nr3d/2026-07-07_09-57-33/ckpt_epoch_207.pth` |

Best NR3D Acc@0.50 in this summary: `EA0+seg (freeze)`, 39.84. Best Mask@0.50: `EA0+seg (freeze)`, 43.51.

## SR3D Accuracy

| Config | Epoch | Acc@0.25 | Acc@0.50 | Mask@0.25 | Mask@0.50 | Source |
|--------|------:|---------:|---------:|----------:|----------:|--------|
| EA0 | 75 | 57.40 | 44.39 | -- | -- | `scripts/experiments/sr3d_fixed3/EA0/sr3d/2026-05-17_21-39-50/ckpt_epoch_75.pth` |
| EA02 | 84 | 55.51 | 43.45 | -- | -- | `scripts/experiments/sr3d_fixed3/EA02/sr3d/2026-05-18_01-12-46/ckpt_epoch_84.pth` |
| EA0+seg (freeze) | 96 | 57.39 | 46.70 | 59.52 | 51.36 | `scripts/experiments/sr3d_fixed3/EA0_SEG_freeze/sr3d/2026-05-25_21-42-09/ckpt_epoch_96.pth` |
| EA02+seg+refine+tps32 unfreeze | 120 | 56.54 | 46.28 | 58.28 | 50.45 | `scripts/experiments/sr3d_fixed3/EA02_SEG_unfreeze_tps32/sr3d/2026-07-07_09-58-18/ckpt_epoch_120.pth` |

Best SR3D Acc@0.50 in this summary: `EA0+seg (freeze)`, 46.70. Best Mask@0.50: `EA0+seg (freeze)`, 51.36.

## ScanRefer Inference Speed Reference

RTX 4090 D, batch_size=1. All rows use warmup=100, 9408 measured samples, and forward-only FPS/memory scope.

| Config | Acc@0.25 | Acc@0.50 | Mask@0.25 | Mask@0.50 | FPS | Peak Alloc (MiB) |
|--------|----------|----------|-----------|-----------|------|-------------------|
| ori | 55.77 | 46.42 | -- | -- | 18.41 | 2049 |
| EA0 | 56.14 | 45.64 | -- | -- | 18.05 | 1116 |
| EA02 | 56.20 | 46.10 | -- | -- | 18.28 | 869 |
| EA0+seg | 55.96 | 48.37 | 58.21 | 51.22 | 12.42 | 1238 |
| EA0+seg+SEGEA | 56.20 | 48.86 | 58.13 | 51.04 | 11.43 | 1381 |
| EA02+seg+refine+tps32 unfreeze | 56.70 | 48.88 | 58.41 | 51.62 | 12.32 | 1048.48 |
| MCLN-single | 51.96 | 33.16 | 55.83 | 47.60 | 6.18 | 972 |
| MCLN-multi | 57.12 | 45.52 | 58.65 | 50.67 | 6.05 | 1000 |
| EDA-single | 53.54 | 41.62 | -- | -- | 10.17 | 806 |
| EDA-multi | 54.45 | 42.31 | -- | -- | 9.84 | 834 |
| BUTD-DETR | 53.70 | 40.90 | -- | -- | 9.14 | 831 |

EDA source log: `/share/lxy/EDA/output/platform_logs/eda_scanrefer_all_eval_20260729_072720.log` (`warmup_iters=100`, `scope=forward_only`, `measured_samples=9408`; multi-stage first, single-stage second).

BUTD-DETR source log: `/share/lxy/butd_detr/output/platform_logs/butd_detr_scanrefer_eval_20260729_111822.log` (`warmup_iters=100`, `scope=forward_only`, `measured_samples=9408`; contrastive alignment accuracy).

EA02+seg+refine+tps32 unfreeze source log: `scripts/experiments/inference_speed_final/EA02_seg_tps32_unfreeze_eval/scanrefer/2026-07-18_06-01-49/log.txt`.
