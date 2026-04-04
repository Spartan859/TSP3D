import argparse
import os
import time

import torch
import torch.distributed as dist


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    return rank, local_rank, world_size


def get_dtype(dtype_name):
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[dtype_name]


def get_flops_per_iter(matrix_size, mode):
    n = float(matrix_size)
    # GEMM FLOPs approximation:
    # forward: 2 * N^3
    # train-like (forward + dA + dB): about 6 * N^3
    if mode == "forward":
        return 2.0 * n * n * n
    return 6.0 * n * n * n


def reduce_metrics(local_tensor):
    if not dist.is_initialized():
        return local_tensor, local_tensor, local_tensor

    world_size = dist.get_world_size()
    mean_tensor = local_tensor.clone()
    dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
    mean_tensor /= world_size

    min_tensor = local_tensor.clone()
    dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)

    max_tensor = local_tensor.clone()
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)

    return mean_tensor, min_tensor, max_tensor


def benchmark_loop(args):
    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    dtype = get_dtype(args.dtype)
    requires_grad = args.mode == "train"

    a = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=dtype, requires_grad=requires_grad)
    b = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=dtype, requires_grad=requires_grad)

    if rank == 0:
        print(
            f"Start benchmark | world_size={world_size} | mode={args.mode} | "
            f"dtype={args.dtype} | matrix_size={args.matrix_size} | interval_sec={args.interval_sec}"
        )

    flops_per_iter = get_flops_per_iter(args.matrix_size, args.mode)

    # Warmup to stabilize kernels/caches.
    for _ in range(max(args.warmup_iters, 0)):
        if args.mode == "train":
            c = torch.matmul(a, b)
            loss = c.sum()
            loss.backward()
            a.grad = None
            b.grad = None
        else:
            with torch.no_grad():
                _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    interval_iters = 0
    interval_total_ms = 0.0
    interval_flops = 0.0
    report_start = time.time()

    try:
        while True:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if args.mode == "train":
                c = torch.matmul(a, b)
                loss = c.sum()
                loss.backward()
                a.grad = None
                b.grad = None
            else:
                with torch.no_grad():
                    _ = torch.matmul(a, b)
            end_event.record()

            torch.cuda.synchronize(device)
            elapsed_ms = start_event.elapsed_time(end_event)

            interval_iters += 1
            interval_total_ms += elapsed_ms
            interval_flops += flops_per_iter

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

            now = time.time()
            interval_wall = now - report_start
            if interval_wall < args.interval_sec:
                continue

            elapsed_s_from_events = interval_total_ms / 1000.0
            steps_per_s = interval_iters / max(elapsed_s_from_events, 1e-9)
            avg_ms = interval_total_ms / max(interval_iters, 1)
            tflops = (interval_flops / max(elapsed_s_from_events, 1e-9)) / 1e12

            local_metrics = torch.tensor([avg_ms, steps_per_s, tflops], device=device)
            mean_m, min_m, max_m = reduce_metrics(local_metrics)

            if rank == 0:
                print(
                    f"[interval {args.interval_sec:.1f}s] "
                    f"avg_ms/iter mean|min|max = {mean_m[0].item():.2f}|{min_m[0].item():.2f}|{max_m[0].item():.2f}, "
                    f"iter/s mean|min|max = {mean_m[1].item():.2f}|{min_m[1].item():.2f}|{max_m[1].item():.2f}, "
                    f"TFLOPS(est) mean|min|max = {mean_m[2].item():.2f}|{min_m[2].item():.2f}|{max_m[2].item():.2f}"
                )

            interval_iters = 0
            interval_total_ms = 0.0
            interval_flops = 0.0
            report_start = time.time()

    except KeyboardInterrupt:
        if rank == 0:
            print("Benchmark stopped by user.")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="GPU matmul occupancy + interval benchmark")
    parser.add_argument("--matrix_size", type=int, default=16384, help="Square matrix size N for N x N matmul")
    parser.add_argument("--interval_sec", type=float, default=5.0, help="Print benchmark metrics every N seconds")
    parser.add_argument("--warmup_iters", type=int, default=20, help="Warmup iterations before reporting")
    parser.add_argument("--mode", choices=["train", "forward"], default="train", help="train: forward+backward, forward: matmul only")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16", help="Compute dtype")
    parser.add_argument("--sleep_ms", type=float, default=0.0, help="Optional sleep between iterations")
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 for fp32 matmul on Ampere+")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark_loop(parse_args())