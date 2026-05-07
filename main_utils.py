# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Shared utilities for all main scripts."""

import argparse
import json
import os, pdb
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.data import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger

from utils import record_tensorboard

from tqdm import tqdm
from get_gt import get_gt
from datetime import datetime 

TQDM_NCOLS = 60

def set_random_seed(seed):
    """Set random seeds for python, numpy and torch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_option():
    """Parse cmd arguments."""
    def str2bool(value):
        if isinstance(value, bool):
            return value
        normalized = value.lower()
        if normalized in ('true', '1', 'yes', 'y', 'on'):
            return True
        if normalized in ('false', '0', 'no', 'n', 'off'):
            return False
        raise argparse.ArgumentTypeError(
            f"Invalid boolean value: {value}. Use true/false."
        )

    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')
    parser.add_argument('--voxel_size', default=0.01, type=float)

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)    # 6
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['sr3d'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='sr3d')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')     # color
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--wo_obj_name', default='None')    # grounding without object name
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--keep_trans_lr", default=4e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument("--box_select_lr", default=4e-4, type=float)
    parser.add_argument("--seg_lr", default=1e-3, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)
    parser.add_argument('--enable_tf32', action='store_true',
                        help='Enable TF32 for matmul/cudnn on Ampere+ GPUs.')
    parser.add_argument('--tf32_matmul', type=str, default='default',
                        choices=['default', 'on', 'off'],
                        help='Control torch.backends.cuda.matmul.allow_tf32 independently.')
    parser.add_argument('--tf32_cudnn', type=str, default='default',
                        choices=['default', 'on', 'off'],
                        help='Control torch.backends.cudnn.allow_tf32 independently.')

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--load_optimizer', action='store_true',
                        help='Load optimizer state from checkpoint')
    parser.add_argument('--load_scheduler', action='store_true',
                        help='Load scheduler state from checkpoint')
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')  # note
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--measure_fps', action='store_true',
                        help='Measure inference FPS during eval (single-card recommended).')
    parser.add_argument('--measure_fps_detail', action='store_true',
                        help='Enable detailed per-stage inference timing (adds profiling overhead).')
    parser.add_argument('--fps_warmup_iters', type=int, default=20,
                        help='Number of warmup eval iterations excluded from FPS stats.')
    parser.add_argument('--fps_max_iters', type=int, default=-1,
                        help='Max eval iterations to include for FPS; -1 means full loader.')
    parser.add_argument('--pp_checkpoint', default=None)    # pointnet checkpoint
    parser.add_argument('--reduce_lr', action='store_true')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True,
                        help='Enable cuDNN benchmark autotune for speed (may reduce reproducibility).')
    parser.add_argument('--cudnn_deterministic', type=str2bool, default=True,
                        help='Enable cuDNN deterministic mode.')
    parser.add_argument('--use_deterministic_algorithms', type=str2bool, default=False,
                        help='Enable torch deterministic algorithms for stricter reproducibility.')
    parser.add_argument('--deterministic_warn_only', action='store_true',
                        help='With --use_deterministic_algorithms, warn instead of raising on nondeterministic ops.')
    parser.add_argument('--use_seg', action='store_true')
    parser.add_argument('--use_refine', action='store_true')
    parser.add_argument('--use_external_attn_bi_layer', type=int, nargs='+', default=[],
                        help='Indices of bi_layers to enable external attention, e.g. 0 1 2')
    parser.add_argument('--use_text_guided_external_attn_bi_layer', type=int, nargs='+', default=[],
                        help='Indices of bi_layers to enable text-guided external attention, e.g. 0 1 2')
    parser.add_argument('--use_film_text_guided_external_attn_bi_layer', type=int, nargs='+', default=[],
                        help='Indices of bi_layers to enable FiLM text-guided external attention, e.g. 0 1 2')
    parser.add_argument('--use_external_attn_bi_layer0', action='store_true')
    parser.add_argument('--use_text_guided_external_attn_bi_layer0', action='store_true')
    parser.add_argument('--use_film_text_guided_external_attn_bi_layer0', action='store_true')
    parser.add_argument('--mink_conv1_stride', type=int, default=2,
                        help='Stride for Minkowski ResNet conv1 (>= 1).')
    parser.add_argument('--use_seg_external_self_attn', dest='use_seg_external_self_attn', action='store_true',
                        help='Enable external self-attention before seg upsample stages.')
    parser.set_defaults(use_seg_external_self_attn=False)
    parser.add_argument('--com_threshold', type=float, default=0.15,
                        help='Threshold for completion branch voxel selection.')
    parser.add_argument('--num_samples_com', type=int, default=2400,
                        help='Number of sampled voxels per scene for completion branch attention.')
    parser.add_argument('--external_attn_coef', type=int, default=4,
                        help='Expansion coefficient used in ExternalMultiheadAttention.')
    parser.add_argument('--external_attn_k_keep0', type=int, default=64,
                        help='EA memory size (k) for keep_trans[0] (bi_layer0). Use 0 for embed_dim//coef.')
    parser.add_argument('--external_attn_k_keep1', type=int, default=64,
                        help='EA memory size (k) for keep_trans[1] (bi_layer1). Use 0 for embed_dim//coef.')
    parser.add_argument('--external_attn_k_com', type=int, default=64,
                        help='EA memory size (k) for com_trans. Use 0 for embed_dim//coef.')
    parser.add_argument('--external_attn_k_seg128', type=int, default=64,
                        help='EA memory size (k) for seg_self_attn_128. Use 0 for embed_dim//coef.')
    parser.add_argument('--external_attn_k_seg64', type=int, default=64,
                        help='EA memory size (k) for seg_self_attn_64. Use 0 for embed_dim//coef.')
    parser.add_argument('--top_pts_threshold', type=int, default=None,
                        help='Top-k candidate points per box for assigner. '
                             'Default: 32 when use_seg=False, 24 when use_seg=True.')
    parser.add_argument('--top_pts_threshold_det', type=int, default=None,
                        help='Top-k candidate points per box in multi-box scenes for assigner. '
                             'Default: 32 when use_seg=False, 8 when use_seg=True.')
    parser.add_argument('--gpu_mem_limit_gb', type=float, default=0.0,
                        help='Per-process GPU memory limit in GiB. 0 disables the limit.')
    parser.add_argument('--prune_threshold_0', type=float, default=0.3,
                        help='Inference pruning threshold for UNet decoder layer 0. '
                             'Points with (1-sigmoid(keep_score)) > threshold are kept.')
    parser.add_argument('--prune_threshold_1', type=float, default=0.7,
                        help='Inference pruning threshold for UNet decoder layer 1.')
    parser.add_argument('--nms_pre', type=int, default=1,
                        help='Top-k candidates before NMS at inference. 1 = take best box directly.')
    parser.add_argument('--nms_iou_thr', type=float, default=0.5,
                        help='IoU threshold for NMS (only used when --nms_pre > 1).')
    parser.add_argument('--nms_score_thr', type=float, default=0.01,
                        help='Score threshold for NMS filtering (only used when --nms_pre > 1).')

    args, _ = parser.parse_known_args()

    if 'wildrefer' in args.dataset or args.test_dataset == 'wildrefer':
        parser.error("Dataset name 'wildrefer' is not supported. Use 'strefer' or 'liferefer'.")

    valid_datasets = {'scanrefer', 'sr3d', 'sr3d+', 'nr3d', 'scannet', 'strefer', 'liferefer'}
    unknown_train_datasets = sorted(set(args.dataset) - valid_datasets)
    if unknown_train_datasets:
        parser.error(
            f"Unsupported dataset(s): {unknown_train_datasets}. "
            f"Valid options: {sorted(valid_datasets)}"
        )
    if args.test_dataset not in valid_datasets:
        parser.error(
            f"Unsupported --test_dataset '{args.test_dataset}'. "
            f"Valid options: {sorted(valid_datasets)}"
        )

    args.eval = args.eval or args.eval_train
    if args.external_attn_coef <= 0:
        parser.error('--external_attn_coef must be a positive integer.')
    if args.top_pts_threshold is None:
        args.top_pts_threshold = 24 if args.use_seg else 32
    if args.top_pts_threshold_det is None:
        args.top_pts_threshold_det = 8 if args.use_seg else 32
    if args.top_pts_threshold <= 0:
        parser.error('--top_pts_threshold must be a positive integer.')
    if args.top_pts_threshold_det <= 0:
        parser.error('--top_pts_threshold_det must be a positive integer.')

    valid_bi_layers = {0, 1, 2}
    args.use_external_attn_bi_layer = sorted(
        set(args.use_external_attn_bi_layer).intersection(valid_bi_layers))
    args.use_text_guided_external_attn_bi_layer = sorted(
        set(args.use_text_guided_external_attn_bi_layer).intersection(valid_bi_layers))
    args.use_film_text_guided_external_attn_bi_layer = sorted(
        set(args.use_film_text_guided_external_attn_bi_layer).intersection(valid_bi_layers))

    if args.use_external_attn_bi_layer0 and 0 not in args.use_external_attn_bi_layer:
        args.use_external_attn_bi_layer.append(0)
    if args.use_text_guided_external_attn_bi_layer0 and 0 not in args.use_text_guided_external_attn_bi_layer:
        args.use_text_guided_external_attn_bi_layer.append(0)
    if args.use_film_text_guided_external_attn_bi_layer0 and 0 not in args.use_film_text_guided_external_attn_bi_layer:
        args.use_film_text_guided_external_attn_bi_layer.append(0)

    # Backward compatibility: --enable_tf32 turns on both controls
    # only when they are not explicitly specified.
    if args.enable_tf32:
        if args.tf32_matmul == 'default':
            args.tf32_matmul = 'on'
        if args.tf32_cudnn == 'default':
            args.tf32_cudnn = 'on'

    args.use_external_attn_bi_layer0 = 0 in args.use_external_attn_bi_layer
    args.use_text_guided_external_attn_bi_layer0 = 0 in args.use_text_guided_external_attn_bi_layer
    args.use_film_text_guided_external_attn_bi_layer0 = 0 in args.use_film_text_guided_external_attn_bi_layer

    return args

# BRIEF load checkpoint.
def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0
    model.load_state_dict(checkpoint['model'], strict=False)
    if not args.eval and not args.reduce_lr:
        if args.load_optimizer and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.load_scheduler and 'scheduler' in checkpoint:
                    # pdb.set_trace()
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                    except Exception as e:
                        print("scheduler loaded failed, maybe due to different scheduler settings.")
                        print(e)
            except Exception as e:
                # pdb.set_trace()
                print("optimizer loaded failed, maybe due to different optimizer settings.")
                print(e)

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()


# BRIEF save model.
def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        
        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split('/')[-1]

        # Use a single run timestamp across all DDP ranks to avoid split log dirs.
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                run_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            else:
                run_time = None
            shared_obj = [run_time]
            dist.broadcast_object_list(shared_obj, src=0)
            current_time = shared_obj[0]
        else:
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            ','.join(args.dataset),
            current_time
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=dist.get_rank(),
            name=name
        )

        # tensorboard
        self.tensorboard = record_tensorboard.TensorBoard(args.log_dir, distributed_rank=dist.get_rank())

        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset


    # BRIEF dataloader.
    def get_loaders(self, args):
        """Initialize data loaders."""
        base_seed = int(args.rng_seed)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        variable_point_keys = {'point_clouds', 'og_color', 'point_instance_label'}

        def _safe_tensorize(item):
            if torch.is_tensor(item):
                return item
            if isinstance(item, np.ndarray):
                # Avoid numpy-backed non-resizable storages in worker-side default_collate.
                return torch.as_tensor(item)
            return item

        def _pad_pointwise(batch_values, fill_value=0):
            tensors = [_safe_tensorize(v) for v in batch_values]
            lengths = [int(t.shape[0]) for t in tensors]
            max_points = max(lengths) if lengths else 0
            out_shape = (len(tensors), max_points) + tuple(tensors[0].shape[1:])
            out = tensors[0].new_full(out_shape, fill_value)
            valid_mask = torch.zeros((len(tensors), max_points), dtype=torch.bool)
            for b, t in enumerate(tensors):
                n = int(t.shape[0])
                out[b, :n] = t
                valid_mask[b, :n] = True
            return out, valid_mask, lengths

        def _pad_gt_masks(batch_values):
            tensors = [_safe_tensorize(v) for v in batch_values]
            num_obj = int(tensors[0].shape[0])
            for t in tensors:
                if t.ndim != 2 or int(t.shape[0]) != num_obj:
                    raise RuntimeError(
                        f"Unexpected gt_masks shape {tuple(t.shape)}; expected [MAX_NUM_OBJ, num_points] "
                        f"with MAX_NUM_OBJ={num_obj}."
                    )
            lengths = [int(t.shape[1]) for t in tensors]
            max_points = max(lengths) if lengths else 0
            out = tensors[0].new_zeros((len(tensors), num_obj, max_points))
            valid_mask = torch.zeros((len(tensors), max_points), dtype=torch.bool)
            for b, t in enumerate(tensors):
                n = int(t.shape[1])
                out[b, :, :n] = t
                valid_mask[b, :n] = True
            return out, valid_mask, lengths

        def safe_collate_fn(batch):
            if len(batch) == 0:
                return {}
            if not isinstance(batch[0], dict):
                return default_collate([_safe_tensorize(item) for item in batch])
            first_keys = set(batch[0].keys())
            for i, sample in enumerate(batch):
                if not isinstance(sample, dict):
                    raise RuntimeError(f"Mixed batch types in collate: got {type(sample)} at index {i}.")
                if set(sample.keys()) != first_keys:
                    raise RuntimeError(
                        f"Inconsistent keys in batch at index {i}. "
                        f"Expected {sorted(first_keys)}, got {sorted(sample.keys())}."
                    )

            collated = {}
            point_valid_mask = None
            point_lengths = None
            for key in batch[0]:
                values = [sample[key] for sample in batch]
                try:
                    if key in variable_point_keys:
                        fill_value = -1 if key == 'point_instance_label' else 0
                        padded, valid_mask, lengths = _pad_pointwise(values, fill_value=fill_value)
                        collated[key] = padded
                        if point_lengths is None:
                            point_lengths = lengths
                            point_valid_mask = valid_mask
                        elif point_lengths != lengths:
                            raise RuntimeError(
                                f"Inconsistent point-wise lengths for key '{key}'. "
                                f"Expected {point_lengths}, got {lengths}."
                            )
                    elif key == 'gt_masks':
                        padded, valid_mask, lengths = _pad_gt_masks(values)
                        collated[key] = padded
                        if point_lengths is None:
                            point_lengths = lengths
                            point_valid_mask = valid_mask
                        elif point_lengths != lengths:
                            raise RuntimeError(
                                f"Inconsistent point-wise lengths for key '{key}'. "
                                f"Expected {point_lengths}, got {lengths}."
                            )
                    else:
                        collated[key] = default_collate([_safe_tensorize(v) for v in values])
                except Exception as e:
                    raise RuntimeError(f"Collate failed for key '{key}': {e}") from e

            if point_valid_mask is not None:
                collated['point_valid_mask'] = point_valid_mask
            return collated

        def seed_worker(worker_id):
            worker_seed = (base_seed + rank * 100000 + worker_id) % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(base_seed + rank)

        if args.eval:
            train_loader = None
        else:
            train_sampler = DistributedSampler(train_dataset, seed=base_seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,      # TODO 
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g,
                collate_fn=safe_collate_fn
            )
        
        test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=base_seed)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g,
            collate_fn=safe_collate_fn
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        losses = ['boxes', 'labels']
        if args.use_contrastive_align:
            losses.append('contrastive_align')
        set_criterion = SetCriterion(
            matcher=matcher,
            losses=losses, eos_coef=0.1, temperature=0.07
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        param_dicts = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "keep_trans" not in n and "text_encoder" not in n
                    and "select" not in n and "seg_unet" not in n and "refine_head" not in n 
                    and "upsample_st" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "keep_trans" in n and p.requires_grad
                ],
                "lr": args.keep_trans_lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "text_encoder" in n and p.requires_grad
                ],
                "lr": args.text_encoder_lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "select" in n and p.requires_grad
                ],
                "lr": args.box_select_lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if ("seg_unet" in n or "upsample_st" in n or "refine_head" in n) and p.requires_grad
                ],
                "lr": args.seg_lr
            }
        ]
        # pdb.set_trace()
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        return optimizer


    # BRIEF main training/testing
    def main(self, args):
        """Run main training/testing pipeline."""
        # Check checkpoint path exists (assert moved earlier than dataset load)
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path), f"Checkpoint path {args.checkpoint_path} does not exist."
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        if not args.eval:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        if not args.eval:
            scheduler = get_scheduler(optimizer, len(train_loader), args)
        else:
            scheduler = None
        
        # Move model to devices
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # synBN
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            else:
                model = model.cuda()

        # note Distributed Data-Parallel Training (DDP)
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=False  , find_unused_parameters=False
        )

        # Check for a checkpoint (file existence asserted earlier)
        if args.checkpoint_path:
            load_checkpoint(args, model, optimizer, scheduler)
        
        # ##############################################
        # NOTE [eval-only] Just eval and end execution #
        # ##############################################
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            return

        # ##############################
        # NOTE Training and Validation #
        # ##############################
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            # train *
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )
            
            # log
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.7f}, '
                'lr_tran {:.7f}, '
                'lr_text {:.7f}, '
                'lr_select {:.7f}, '
                'lr_seg {:.7f}'.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr'],
                    optimizer.param_groups[2]['lr'],
                    optimizer.param_groups[3]['lr'],
                    optimizer.param_groups[4]['lr']
                )
            )

            # save model and validate
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0:
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                
                # validate *
                print("Test evaluation.......")
                self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )

        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        # Training is over (only rank0 writes checkpoint files).
        if dist.get_rank() == 0:
            save_checkpoint(args, 'last', model, optimizer, scheduler, True)
            self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            'target_cat': batch_data['target_cat']
        }
        if 'point_valid_mask' in batch_data:
            inputs['point_valid_mask'] = batch_data['point_valid_mask']
        return inputs
        
    @staticmethod
    def _get_inputs_contra(batch_data):
        gt_labels = batch_data['sem_cls_label']
        gt_center = batch_data['center_label'][:, :, 0:3]
        gt_size = batch_data['size_gts']
        gt_bbox = torch.cat([gt_center, gt_size], dim=-1) 
        positive_map = batch_data['positive_map']               # main obj.
        modify_positive_map = batch_data['modify_positive_map'] # attribute(modify)
        pron_positive_map = batch_data['pron_positive_map']     # pron
        other_entity_map = batch_data['other_entity_map']       # other(auxi)
        rel_positive_map = batch_data['rel_positive_map']       # relation
        box_label_mask = batch_data['box_label_mask'] 
        target = [
            {
                "boxes": gt_bbox[b, box_label_mask[b].bool()],
                "positive_map": positive_map[b, box_label_mask[b].bool()],
                "modify_positive_map": modify_positive_map[b, box_label_mask[b].bool()],
                "pron_positive_map": pron_positive_map[b, box_label_mask[b].bool()],
                "other_entity_map": other_entity_map[b, box_label_mask[b].bool()],
                "rel_positive_map": rel_positive_map[b, box_label_mask[b].bool()]
            }
            for b in range(gt_labels.shape[0])
        ]       
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            'target':target
        }
        if 'point_valid_mask' in batch_data:
            inputs['point_valid_mask'] = batch_data['point_valid_mask']
        return inputs

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk
        )
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict

    @staticmethod
    def _tqdm_newline(progress_bar):
        # Ensure logger output starts on a fresh line instead of sharing tqdm's dynamic line.
        if progress_bar is not None:
            progress_bar.write("")


    # BRIEF Training
    def train_one_epoch(self, epoch, train_loader, model,
                        criterion, set_criterion,
                        optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode

        # Loop over batches
        train_loader = tqdm(train_loader, ncols=TQDM_NCOLS)
        for batch_idx, batch_data in enumerate(train_loader):
            gt_bboxes_3d, gt_labels_3d, gt_all_bbox_new, auxi_bbox, gt_masks, img_metas = get_gt(batch_data)
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            # get the input data: pointcloud and text
            inputs = self._get_inputs(batch_data)
            
            losses = model(inputs, gt_bboxes_3d, gt_labels_3d, gt_all_bbox_new, auxi_bbox, gt_masks, img_metas, epoch)
            loss = losses['loss']
            if not torch.isfinite(loss):
                self._tqdm_newline(train_loader)
                self.logger.warning(
                    f"Skip non-finite loss at epoch {epoch}, iter {batch_idx + 1}: "
                    + ', '.join(
                        f"{k}={float(v.detach().cpu()) if torch.is_tensor(v) and v.numel() == 1 else v}"
                        for k, v in losses.items() if 'loss' in k
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()

            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
                if not torch.isfinite(grad_total_norm):
                    self._tqdm_newline(train_loader)
                    self.logger.warning(
                        f"Skip optimizer step due to non-finite grad norm at "
                        f"epoch {epoch}, iter {batch_idx + 1}: grad_norm={grad_total_norm}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                stat_dict['grad_norm'] = grad_total_norm
            
            optimizer.step()
            scheduler.step()

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, losses)

            # print loss
            if (batch_idx + 1) % args.print_freq == 0:
                self._tqdm_newline(train_loader)
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '  # Train: [30][2000/2432]
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / (batch_idx + 1):.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key
                ])) # loss，loss_bbox，loss_ce，loss_sem_align，loss_giou，query_points_generation_loss


    # BRIEF eval 
    @staticmethod
    @torch.no_grad()
    def _main_eval_branch(batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args, logger=None):
        # Move to GPU
        gt_bboxes_3d, gt_labels_3d, gt_all_bbox_new, auxi_bbox, gt_masks, img_metas = get_gt(batch_data)
        batch_data = BaseTrainTester._to_gpu(batch_data)
        inputs = BaseTrainTester._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # STEP Forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        bbox_results, seg_masks, losses, backbone_time, trans_time = model(
            inputs, gt_bboxes_3d, gt_labels_3d, gt_all_bbox_new, auxi_bbox, gt_masks, img_metas=img_metas)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        inf_time = end_time - start_time

        end_points = {'bbox_results': bbox_results, 'gt_bboxes_3d': gt_bboxes_3d, "seg_pred": seg_masks, "seg_gt": gt_masks}
        # STEP Compute loss
        for key in batch_data:
            assert (key not in end_points)
            end_points[key] = batch_data[key]

        stat_dict = BaseTrainTester._accumulate_stats(stat_dict, losses)
        if (batch_idx + 1) % args.print_freq == 0 and logger is not None:
            BaseTrainTester._tqdm_newline(test_loader)
            logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key
            ]))
        return stat_dict, end_points, inf_time, backbone_time, trans_time

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
