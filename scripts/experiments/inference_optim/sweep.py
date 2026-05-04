"""
Inference parameter sweep: loads dataset + model once per GPU worker, then
loops over its assigned parameter combinations in-process by patching
model.head attributes.

When --gpu_ids lists multiple GPUs, the param_list is distributed round-robin
across workers; each worker runs independently (single-card, world_size=1).

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/experiments/inference_optim/sweep.py \
        --checkpoint_path /path/to/ckpt.pth \
        --data_root /root/lxy/TSP3D/data/ \
        --use_external_attn_bi_layer 0 2 \
        --use_film_text_guided_external_attn_bi_layer 0 2 \
        --sweep_params '{"com_threshold":[0.05,0.10,0.15,0.20,0.30]}' \
        --sweep_mode grid \
        --output_jsonl /tmp/sweep_results.jsonl

Usage (multi-GPU, auto-distributed):
    python scripts/experiments/inference_optim/sweep.py \
        --gpu_ids 0,1,2,3 \
        --checkpoint_path /path/to/ckpt.pth \
        ...same flags...
"""

import argparse
import itertools
import json
import os
import sys
import time
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


def _get_head(model):
    m = model.module if hasattr(model, 'module') else model
    return m.head


def _patch_head(head, params):
    if 'com_threshold' in params:
        head.com_threshold = float(params['com_threshold'])
    if 'num_samples_com' in params:
        head.num_samples_com = int(params['num_samples_com'])
    if 'prune_threshold_0' in params or 'prune_threshold_1' in params:
        t0 = params.get('prune_threshold_0', head.prune_threshold[0])
        t1 = params.get('prune_threshold_1', head.prune_threshold[1])
        head.prune_threshold = (float(t0), float(t1))
    if 'nms_pre' in params or 'nms_iou_thr' in params or 'nms_score_thr' in params:
        head.test_cfg = dict(
            nms_pre=int(params.get('nms_pre', head.test_cfg['nms_pre'])),
            iou_thr=float(params.get('nms_iou_thr', head.test_cfg['iou_thr'])),
            score_thr=float(params.get('nms_score_thr', head.test_cfg['score_thr'])),
        )


def run_eval(model, test_loader, args, train_tester):
    # Import torch lazily here so module-level import doesn't initialize CUDA
    torch = importlib.import_module('torch')
    with torch.no_grad():
        metrics = train_tester.evaluate_grounding(test_loader, model, args)
    return {k: v for k, v in metrics.items() if not k.startswith('_')}


def build_param_grid(sweep_params, sweep_mode):
    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    if sweep_mode == 'zip':
        combos = list(zip(*values))
    else:
        combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def _distribute_round_robin(param_list, n_workers):
    """Split param_list into n_workers sublists using round-robin assignment."""
    buckets = [[] for _ in range(n_workers)]
    for i, p in enumerate(param_list):
        buckets[i % n_workers].append(p)
    return buckets


def parse_sweep_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--sweep_params', type=str, default='{}')
    p.add_argument('--sweep_mode', choices=['grid', 'zip'], default='grid')
    p.add_argument('--output_jsonl', type=str, default='sweep_results.jsonl')
    p.add_argument('--batch_size_eval', type=int, default=1)
    p.add_argument('--gpu_ids', type=str, default='',
                   help='Comma-separated GPU ids to use, e.g. "0,1,2,3". '
                        'If empty, uses CUDA_VISIBLE_DEVICES / GPU 0.')
    sweep_args, remaining = p.parse_known_args()
    return sweep_args, remaining


def _worker(rank, gpu_id, param_sublist, output_jsonl, remaining_argv, sweep_args):
    """Single-GPU worker: loads data+model once, runs its param_sublist."""
    os.environ['RANK']        = '0'
    os.environ['LOCAL_RANK']  = '0'
    os.environ['WORLD_SIZE']  = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # Use different ports per worker to avoid conflicts
    os.environ['MASTER_PORT'] = str(29500 + rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import torch and related modules after setting CUDA_VISIBLE_DEVICES so CUDA sees the mapping
    torch = importlib.import_module('torch')
    dist = importlib.import_module('torch.distributed')
    from torch.utils.data import DataLoader, SequentialSampler
    main_utils = importlib.import_module('main_utils')
    train_dist_mod = importlib.import_module('train_dist_mod')
    parse_option = main_utils.parse_option
    load_checkpoint = main_utils.load_checkpoint
    train_tester = train_dist_mod.TrainTester

    sys.argv = [sys.argv[0]] + remaining_argv + ['--eval']
    args = parse_option()
    args.eval = True
    args.local_rank = 0

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(0)  # always 0 inside this worker (CUDA_VISIBLE_DEVICES remaps)

    tag = f'[GPU {gpu_id}]'
    print(f'{tag} Loading dataset...')
    t0 = time.time()
    _, test_dataset = train_tester.get_datasets(args)
    sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=sweep_args.batch_size_eval,
        sampler=sampler,
        num_workers=min(4, args.num_workers),
        collate_fn=getattr(test_dataset, 'collate_fn', None),
        pin_memory=True,
    )
    print(f'{tag} Dataset loaded in {time.time()-t0:.1f}s, {len(test_dataset)} samples.')

    print(f'{tag} Loading model...')
    t0 = time.time()
    model = train_tester.get_model(args).cuda()
    from torch.nn.parallel import DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[0],
                                    broadcast_buffers=False,
                                    find_unused_parameters=False)
    load_checkpoint(args, model, None, None)
    model.eval()
    print(f'{tag} Model loaded in {time.time()-t0:.1f}s.')

    head = _get_head(model)

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    results = []
    with open(output_jsonl, 'w') as fout:
        for i, params in enumerate(param_sublist):
            _patch_head(head, params)
            print(f'\n{tag} Run {i+1}/{len(param_sublist)}: {params}')
            metrics = run_eval(model, test_loader, args, train_tester)
            record = {**params, **metrics}
            results.append(record)
            fout.write(json.dumps(record) + '\n')
            fout.flush()
            print(f'{tag}   → Acc0.25={metrics["acc0.25"]:.5f}  '
                  f'Acc0.50={metrics["acc0.50"]:.5f}  '
                  f'lat={metrics["avg_latency_ms"]:.1f}ms  '
                  f'fps={metrics["fps"]:.2f}  '
                  f'peak_mem={metrics.get("mem_peak_res_mib", float("nan")):.0f}MiB  '
                  f'({metrics["elapsed_s"]:.1f}s total)')

    print(f'{tag} Done. Results written to {output_jsonl}')


def main():
    sweep_args, remaining = parse_sweep_args()

    sweep_params = json.loads(sweep_args.sweep_params)
    param_list = build_param_grid(sweep_params, sweep_args.sweep_mode) if sweep_params else [{}]
    print(f'[sweep] {len(param_list)} parameter combination(s) to evaluate.')

    # Resolve GPU list
    if sweep_args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in sweep_args.gpu_ids.split(',') if g.strip()]
    else:
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cvd:
            gpu_ids = [int(g.strip()) for g in cvd.split(',') if g.strip()]
        else:
            gpu_ids = [0]

    n_workers = min(len(gpu_ids), len(param_list))
    gpu_ids = gpu_ids[:n_workers]
    print(f'[sweep] Using {n_workers} GPU(s): {gpu_ids}')

    buckets = _distribute_round_robin(param_list, n_workers)
    for i, (gid, bucket) in enumerate(zip(gpu_ids, buckets)):
        print(f'  GPU {gid}: {len(bucket)} run(s)')

    output_base = sweep_args.output_jsonl
    stem, ext = os.path.splitext(output_base)

    if n_workers == 1:
        # Single-worker: run in-process (simpler, no spawn overhead)
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('LOCAL_RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        if sweep_args.gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        _worker(0, gpu_ids[0], buckets[0], output_base, remaining, sweep_args)
        shard_files = [output_base]
    else:
        shard_files = [f'{stem}_gpu{gid}{ext}' for gid in gpu_ids]
        processes = []
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        for rank, (gid, bucket, shard) in enumerate(zip(gpu_ids, buckets, shard_files)):
            p = ctx.Process(
                target=_worker,
                args=(rank, gid, bucket, shard, remaining, sweep_args),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            print(f'[sweep] WARNING: {len(failed)} worker(s) failed.')

        # Merge shards in original param_list order
        print(f'\n[sweep] Merging {len(shard_files)} shard(s) into {output_base}...')
        # Build lookup: param key tuple -> record
        all_records = {}
        for shard in shard_files:
            if not os.path.exists(shard):
                continue
            with open(shard) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        # key = tuple of sweep param values in original order
                        key = tuple(rec.get(k) for k in sweep_params)
                        all_records[key] = rec

        os.makedirs(os.path.dirname(os.path.abspath(output_base)), exist_ok=True)
        with open(output_base, 'w') as fout:
            for params in param_list:
                key = tuple(params.get(k) for k in sweep_params)
                if key in all_records:
                    fout.write(json.dumps(all_records[key]) + '\n')

    # Summary table
    results = []
    with open(output_base) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    results.sort(key=lambda r: r.get('acc0.25', 0), reverse=True)
    print('\n[sweep] Results sorted by Acc0.25:')
    param_keys = list(sweep_params.keys())
    summary_cols = ['acc0.25', 'acc0.50', 'avg_latency_ms', 'fps',
                    'mem_peak_res_mib', 'elapsed_s']
    header = param_keys + summary_cols
    col_w = max(14, max((len(h) for h in header), default=14))
    print('  ' + '  '.join(h.ljust(col_w) for h in header))
    print('  ' + '  '.join('-' * col_w for _ in header))
    for r in results:
        row = [str(r.get(k, '')) for k in param_keys]
        row += [str(r.get(c, '')) for c in summary_cols]
        print('  ' + '  '.join(v.ljust(col_w) for v in row))

    print(f'\n[sweep] Results written to {output_base}')


if __name__ == '__main__':
    main()
