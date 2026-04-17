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
"""Main script for language modulation."""

import os
import random
import numpy as np
import torch
import torch.distributed as dist

from main_utils import parse_option, BaseTrainTester, set_random_seed
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths

from tqdm import tqdm
import datetime

import ipdb
st = ipdb.set_trace

import numpy as np

class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    # BRIEF Initialize dataset.
    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""

        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        if args.joint_det:
            dataset_dict['scannet'] = 10
        print('Loading datasets:', sorted(list(dataset_dict.keys())))

        if args.eval:
            train_dataset = None
        else:
            train_dataset = Joint3DDataset(
                dataset_dict=dataset_dict,
                test_dataset=args.test_dataset,
                split='train' if not args.debug else 'val',
                use_color=args.use_color, use_height=args.use_height,
                overfit=args.debug,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=args.augment_det
            )
        
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_train else 'train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            wo_obj_name=args.wo_obj_name
        )
        return train_dataset, test_dataset

    # BRIEF Initialize the model.
    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            data_path = args.data_root,
            self_attend=args.self_attend,
            voxel_size = args.voxel_size,
            use_seg=args.use_seg,
            use_refine=args.use_refine,
            use_seg_external_self_attn=args.use_seg_external_self_attn,
            use_external_attn_bi_layer=args.use_external_attn_bi_layer,
            use_text_guided_external_attn_bi_layer=args.use_text_guided_external_attn_bi_layer,
            use_film_text_guided_external_attn_bi_layer=args.use_film_text_guided_external_attn_bi_layer,
            com_threshold=args.com_threshold,
            num_samples_com=args.num_samples_com,
            external_attn_coef=args.external_attn_coef,
            mink_conv1_stride=args.mink_conv1_stride
        )
        return model

    # BRIEF input data.
    @staticmethod
    def _get_inputs(batch_data):
        # print(batch_data['utterances'])
        return {
            'point_clouds': batch_data['point_clouds'].float(), # ([B, 50000, 6]) xyz + colour
            'text': batch_data['utterances'],                   # list[B]  text
            'target_cat': batch_data['target_cat']
        }


    # BRIEF only eval one epoch.
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        # [Option] Object detection evaluation on ScanNet dataset.
        if args.test_dataset == 'scannet':      
            return self.evaluate_one_epoch_det(
                epoch, test_loader, model,
                criterion, set_criterion, args
            )

        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        prefixes = ['3dcnn']

        evaluator = GroundingEvaluator(
            only_root=True, thresholds=[0.25, 0.5],     
            topks=[1], prefixes=prefixes,
            filter_non_gt_boxes=args.butd_cls,
            use_seg=args.use_seg
        )

        # NOTE Main eval branch
        test_loader = tqdm(test_loader)
        inf_speeds, vis_back_speeds, text_back_speeds, fuiosn_speeds, head_speeds = [],[],[],[],[]
        ext_bi0_speeds, ext_bi1_speeds, ext_bi2_speeds, ext_total_speeds = [], [], [], []
        fps_enabled = bool(getattr(args, 'measure_fps', False))
        fps_warmup_iters = max(int(getattr(args, 'fps_warmup_iters', 20)), 0)
        fps_max_iters = int(getattr(args, 'fps_max_iters', -1))
        total_fps_samples = 0
        total_fps_time = 0.0
        total_ext_module_time = 0.0
        mem_allocated_mb = []
        mem_reserved_mb = []
        max_allocated_mb = 0.0
        max_reserved_mb = 0.0
        mem_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if mem_device is not None:
            torch.cuda.reset_peak_memory_stats(mem_device)
        for batch_idx, batch_data in enumerate(test_loader):
            if fps_max_iters > 0 and batch_idx >= fps_max_iters:
                break
            # note forward and compute loss
            stat_dict, end_points, inf_speed, backbone_time, detail_time  = self._main_eval_branch(     
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            inf_speeds.append(inf_speed)
            vis_back_speeds.append(detail_time[0])
            text_back_speeds.append(detail_time[1])
            fuiosn_speeds.append(detail_time[2])
            head_speeds.append(detail_time[3])
            head_module = model.module.head if hasattr(model, 'module') else model.head
            ext_profile = getattr(head_module, 'last_external_attn_profile', None)
            if ext_profile is None:
                ext_profile = {'bi_layer0': 0.0, 'bi_layer1': 0.0, 'bi_layer2': 0.0, 'total': 0.0}
            ext_bi0_speeds.append(float(ext_profile.get('bi_layer0', 0.0)))
            ext_bi1_speeds.append(float(ext_profile.get('bi_layer1', 0.0)))
            ext_bi2_speeds.append(float(ext_profile.get('bi_layer2', 0.0)))
            ext_total_speeds.append(float(ext_profile.get('total', 0.0)))
            if fps_enabled and batch_idx >= fps_warmup_iters:
                batch_size = int(batch_data['point_clouds'].shape[0])
                total_fps_samples += batch_size
                total_fps_time += float(inf_speed)
                total_ext_module_time += float(ext_profile.get('total', 0.0))
            if mem_device is not None and batch_idx >= fps_warmup_iters:
                cur_allocated_mb = torch.cuda.memory_allocated(mem_device) / (1024.0 ** 2)
                cur_reserved_mb = torch.cuda.memory_reserved(mem_device) / (1024.0 ** 2)
                cur_max_allocated_mb = torch.cuda.max_memory_allocated(mem_device) / (1024.0 ** 2)
                cur_max_reserved_mb = torch.cuda.max_memory_reserved(mem_device) / (1024.0 ** 2)
                mem_allocated_mb.append(cur_allocated_mb)
                mem_reserved_mb.append(cur_reserved_mb)
                max_allocated_mb = max(max_allocated_mb, cur_max_allocated_mb)
                max_reserved_mb = max(max_reserved_mb, cur_max_reserved_mb)
            if evaluator is not None:
                for prefix in prefixes:
                    # note only consider the last layer
                    if prefix != '3dcnn':
                        continue

                    # evaluation
                    evaluator.evaluate(end_points, prefix)      

        evaluator.synchronize_between_processes()
        if dist.get_rank() == 0:
            if evaluator is not None:

                evaluator.print_stats()
                self.logger.info(f'Eval: [{epoch}]  ')
                for t in evaluator.thresholds:
                    self.logger.info(''.join([
                        f"{'3dcnn'} Acc{t:.2f}: ", f"Top-{1}: {evaluator.dets[('3dcnn', t, 1, 'bbf')] / max(evaluator.gts[('3dcnn', t, 1, 'bbf')], 1):.5f}"
                    ]))   
                if args.use_seg:        
                    self.logger.info('Acc_mask0.25' + ' ' +  str(evaluator.dets['overall_mask'] / evaluator.gts['mask_3dcnn']))  
                    self.logger.info('Acc_mask0.50' + ' ' +  str(evaluator.dets['overall50_mask'] / evaluator.gts['mask_3dcnn']))
            print('inf: ', np.array(inf_speeds).mean(),'vis_back_speeds: ', np.array(vis_back_speeds).mean(),
                'text_back_speeds: ', np.array(text_back_speeds).mean(),'fuiosn_speeds: ', np.array(fuiosn_speeds).mean(),
                'head_speeds: ', np.array(head_speeds).mean())
            self.logger.info(
                'External-attn-affected module time(s): '
                f'bi_layer0={np.array(ext_bi0_speeds).mean():.6f}, '
                f'bi_layer1={np.array(ext_bi1_speeds).mean():.6f}, '
                f'bi_layer2={np.array(ext_bi2_speeds).mean():.6f}, '
                f'total={np.array(ext_total_speeds).mean():.6f}'
            )
            if len(mem_allocated_mb) > 0:
                self.logger.info(
                    'GPU memory(MiB): '
                    f'avg_allocated={np.mean(mem_allocated_mb):.2f}, '
                    f'avg_reserved={np.mean(mem_reserved_mb):.2f}, '
                    f'peak_allocated={max_allocated_mb:.2f}, '
                    f'peak_reserved={max_reserved_mb:.2f}, '
                    f'warmup_iters={fps_warmup_iters}'
                )
            elif mem_device is not None:
                self.logger.info(
                    'GPU memory(MiB): N/A (no measured iterations; reduce --fps_warmup_iters or increase eval iters).'
                )
            if fps_enabled:
                if total_fps_samples > 0 and total_fps_time > 0:
                    fps = total_fps_samples / total_fps_time
                    avg_latency_ms = (total_fps_time / total_fps_samples) * 1000.0
                    self.logger.info(
                        f'FPS(single-card): {fps:.3f} | Avg latency: {avg_latency_ms:.3f} ms/sample '
                        f'| warmup_iters={fps_warmup_iters} | measured_samples={total_fps_samples}'
                    )
                    if total_ext_module_time > 0:
                        ext_fps = total_fps_samples / total_ext_module_time
                        ext_ratio = total_ext_module_time / total_fps_time
                        self.logger.info(
                            f'External-attn-affected path FPS(eqv): {ext_fps:.3f} '
                            f'| time_ratio={ext_ratio:.4f} of end-to-end measured inference'
                        )
                else:
                    self.logger.info(
                        'FPS(single-card): N/A (no measured samples; reduce --fps_warmup_iters or increase eval iters).'
                    )

        return None
       
    # BRIEF Scannet detection evalution
    @torch.no_grad()
    def evaluate_one_epoch_det(self, epoch, test_loader,
                               model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        if set_criterion is not None:
            set_criterion.eval()

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes += [
                f'{i}head_' for i in range(args.num_decoder_layers - 1)
            ]
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes = ['last_']
        ap_calculator_list = [
            APCalculator(iou_thresh, dataset_config.class2type)
            for iou_thresh in args.ap_iou_thresholds
        ]
        mAPs = [
            [iou_thresh, {k: 0 for k in prefixes}]
            for iou_thresh in args.ap_iou_thresholds
        ]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        # NOTE char span and token span.
        wordidx = np.array([
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        ])  # 18+1（not mentioned）
        tokenidx = np.array([
            1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        ])  # 18 token span

        test_loader = tqdm(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            # note eval
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )

            # step score   contrast
            proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores  # ([B, 256, 256])

            # step
            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19] # ([B, 256, 19])
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls     # ([B, 256, 19])

            # step Parse predictions
            # for prefix in prefixes:
            prefix = 'last_'
            # pred
            batch_pred_map_cls = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)
            batch_gt_map_cls = parse_groundtruths(
                end_points, CONFIG_DICT,
                size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        # for prefix in prefixes:
        prefix = 'last_'
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(
                batch_pred_map_cls_dict[prefix],
                batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.logger.info(
                '=====================>'
                f'{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}'
                '<====================='
            )
            for key in metrics_dict:
                self.logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

        for mAP in mAPs:
            self.logger.info(
                f'IoU[{mAP[0]}]:\t'
                + ''.join([
                    f'{key}: {mAP[1][key]:.4f} \t'
                    for key in sorted(mAP[1].keys())
                ])
            )

        return None


if __name__ == '__main__':
    # huggingface
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    opt = parse_option()

    if opt.use_deterministic_algorithms and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # Required by some CUDA/cuBLAS paths for deterministic GEMM behavior.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if opt.local_rank is None:
        opt.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print("LOCAL_RANK", opt.local_rank)
    
    # distributed 
    torch.cuda.set_device(opt.local_rank)
    # https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1304721237
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))  
    set_random_seed(opt.rng_seed + opt.local_rank)
    
    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = bool(opt.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(opt.cudnn_deterministic)

    if opt.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=opt.deterministic_warn_only)

    if opt.tf32_matmul != 'default':
        torch.backends.cuda.matmul.allow_tf32 = (opt.tf32_matmul == 'on')
        _set_float32_matmul_precision = getattr(torch, 'set_float32_matmul_precision', lambda *args, **kwargs: None)
        if opt.tf32_matmul == 'on':
            _set_float32_matmul_precision('high')
        else:
            _set_float32_matmul_precision('highest')

    if opt.tf32_cudnn != 'default':
        torch.backends.cudnn.allow_tf32 = (opt.tf32_cudnn == 'on')

    if dist.get_rank() == 0:
        print(
            'TF32 config -> '
            f'enable_tf32_arg={opt.enable_tf32}, '
            f'tf32_matmul_arg={opt.tf32_matmul}, '
            f'tf32_cudnn_arg={opt.tf32_cudnn}, '
            f'use_deterministic_algorithms={opt.use_deterministic_algorithms}, '
            f'deterministic_warn_only={opt.deterministic_warn_only}, '
            f'cudnn.benchmark={torch.backends.cudnn.benchmark}, '
            f'cudnn.deterministic={torch.backends.cudnn.deterministic}, '
            f'matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, '
            f'cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, '
            f'float32_matmul_precision={getattr(torch, "get_float32_matmul_precision", lambda: "N/A")()}'
        )
        if torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic:
            print('Warning: both cudnn.benchmark=True and cudnn.deterministic=True are enabled; this may reduce reproducibility and can hurt performance predictability.')

    train_tester = TrainTester(opt)
    ckpt_path = train_tester.main(opt)
