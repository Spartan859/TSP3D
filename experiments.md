## inference speed

| exp | fps | avg_latency_ms | avg_allocated_mib | avg_reserved_mib | peak_allocated_mib | peak_reserved_mib |
|---|---:|---:|---:|---:|---:|---:|
| test_ori | 15.906 | 62.868 | 669.33 | 1594.47 | 2072.99 | 8204.00 |
| test_EA0 | 16.821 | 59.449 | 671.12 | 1276.50 | 1143.60 | 1504.00 |
| test_EA02_num_sps_com_2400 | 15.784 | 63.354 | 673.31 | 938.76 | 947.65 | 1160.00 |
| test_EA02_num_sps_com_3000 | 15.310 | 65.317 | 672.91 | 925.91 | 946.83 | 1152.00 |

| exp | bi_layer0_s | bi_layer1_s | bi_layer2_s | ext_total_s | ext_path_fps_eqv | ext_time_ratio |
|---|---:|---:|---:|---:|---:|---:|
| test_ori | 0.008149 | 0.004016 | 0.004939 | 0.017105 | 58.697 | 0.2710 |
| test_EA0 | 0.007557 | 0.003777 | 0.005018 | 0.016351 | 61.396 | 0.2740 |
| test_EA02_num_sps_com_2400 | 0.008625 | 0.004024 | 0.005131 | 0.017781 | 56.428 | 0.2797 |
| test_EA02_num_sps_com_3000 | 0.008793 | 0.004174 | 0.005491 | 0.018458 | 54.371 | 0.2816 |

来源日志：

- scripts/experiments/inference_speed/test_ori/scanrefer/2026-04-08_17-31-52/log.txt
- scripts/experiments/inference_speed/test_EA0/scanrefer/2026-04-08_17-27-07/log.txt
- scripts/experiments/inference_speed/test_EA02_num_sps_com_2400/scanrefer/2026-04-08_17-47-32/log.txt
- scripts/experiments/inference_speed/test_EA02_num_sps_com_3000/scanrefer/2026-04-08_17-49-54/log.txt


