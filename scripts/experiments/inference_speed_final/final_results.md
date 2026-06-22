# Inference Speed Final — ScanRefer Results

RTX 4090 D, batch_size=1. TSP3D rows use warmup=100, 9408 samples. EDA and BUTD-DETR rows use warmup=20, 9488 samples.

| Config | Acc@0.25 | Acc@0.50 | Mask@0.25 | Mask@0.50 | FPS | Peak Alloc (MiB) |
|--------|----------|----------|-----------|-----------|------|-------------------|
| ori | 55.77 | 46.42 | -- | -- | 18.41 | 2049 |
| EA0 | 56.14 | 45.64 | -- | -- | 18.05 | 1116 |
| EA02 | 56.20 | 46.10 | -- | -- | 18.28 | 869 |
| EA0+seg | 55.96 | 48.37 | 58.21 | 51.22 | 12.42 | 1238 |
| EA0+seg+SEGEA | 56.20 | 48.86 | 58.13 | 51.04 | 11.43 | 1381 |
| MCLN-single | 51.96 | 33.16 | 55.83 | 47.60 | 6.18 | 972 |
| MCLN-multi | 57.12 | 45.52 | 58.65 | 50.67 | 6.05 | 1000 |
| EDA-single | 53.51 | 41.61 | -- | -- | 9.55 | 17463 |
| EDA-multi | 54.45 | 42.30 | -- | -- | 9.16 | 17491 |
| BUTD-DETR | 53.70 | 40.90 | -- | -- | 9.05 | 17489 |

EDA source log: `/mnt/share/algorithm/kimi/cache/lxy/EDA/outputs/platform_logs/eda_scanrefer_all_eval_20260622_084340.log`.

BUTD-DETR source log: `/mnt/share/algorithm/kimi/cache/lxy/butd_detr/outputs/platform_logs/butd_detr_scanrefer_eval_20260622_114716.log`.
