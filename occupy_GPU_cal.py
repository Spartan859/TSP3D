import torch
import torch.distributed as dist
import os
import time

def heavy_matrix_op(rank, world_size, matrix_size):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"[Rank {rank}] Using device: {device}")

    a = torch.randn((matrix_size, matrix_size), device=device, requires_grad=True)
    b = torch.randn((matrix_size, matrix_size), device=device, requires_grad=True)

    while True:
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        # 清理显存
        a.grad = None
        b.grad = None
        # Optional: torch.cuda.synchronize()
        # time.sleep(0.01)

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    matrix_size = 16384  # A100 可以撑很大, 可调 16k-24k
    # torchrun/torch.distributed.launch 自动设置以下环境变量
    rank = int(os.environ.get("LOCAL_RANK", 0))

    heavy_matrix_op(rank, world_size, matrix_size)