## ori
```
[04/20 16:01:45] ori_eval_4G INFO: Eval: [61]  
[04/20 16:01:45] ori_eval_4G INFO: 3dcnn Acc0.25: Top-1: 0.55269
[04/20 16:01:45] ori_eval_4G INFO: 3dcnn Acc0.50: Top-1: 0.45067
[04/20 16:01:45] ori_eval_4G INFO: External-attn-affected module time(s): bi_layer0=0.005429, bi_layer1=0.005457, bi_layer2=0.004297, total=0.015184
[04/20 16:01:45] ori_eval_4G INFO: GPU memory(MiB): avg_allocated=637.74, avg_reserved=3184.84, peak_allocated=2019.92, peak_reserved=4078.00, warmup_iters=100
[04/20 16:01:45] ori_eval_4G INFO: FPS(single-card): 15.515 | Avg latency: 64.455 ms/sample | warmup_iters=100 | measured_samples=9408
[04/20 16:01:45] ori_eval_4G INFO: External-attn-affected path FPS(eqv): 65.883 | time_ratio=0.2355 of end-to-end measured inference
```

## EA0
```
[04/20 16:14:03] EA0_eval_4G INFO: 3dcnn Acc0.25: Top-1: 0.56111
[04/20 16:14:03] EA0_eval_4G INFO: 3dcnn Acc0.50: Top-1: 0.45635
[04/20 16:14:03] EA0_eval_4G INFO: External-attn-affected module time(s): bi_layer0=0.005075, bi_layer1=0.005738, bi_layer2=0.004371, total=0.015184
[04/20 16:14:03] EA0_eval_4G INFO: GPU memory(MiB): avg_allocated=639.53, avg_reserved=1262.48, peak_allocated=1108.60, peak_reserved=1264.00, warmup_iters=100
[04/20 16:14:03] EA0_eval_4G INFO: FPS(single-card): 15.397 | Avg latency: 64.947 ms/sample | warmup_iters=100 | measured_samples=9408
[04/20 16:14:03] EA0_eval_4G INFO: External-attn-affected path FPS(eqv): 65.905 | time_ratio=0.2336 of end-to-end measured inference
```

## EA02