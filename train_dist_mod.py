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
from src.grounding_evaluator import GroundingEvaluator, scores_to_box_scores
from src.wildrefer_official_eval import cal_accuracy as wildrefer_cal_accuracy
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths

from tqdm import tqdm
import datetime

import ipdb
st = ipdb.set_trace

import numpy as np

TQDM_NCOLS = 60

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

        wildrefer_dsets = {'strefer', 'liferefer'}
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        if args.test_dataset in wildrefer_dsets and (
            len(dataset_dict) != 1 or args.test_dataset not in dataset_dict
        ):
            raise ValueError(
                "For strefer/liferefer runs, use matching single-dataset args: "
                "--dataset <strefer|liferefer> --test_dataset <same>."
            )
        selected_wildrefer = [d for d in dataset_dict if d in wildrefer_dsets]
        if selected_wildrefer and (len(dataset_dict) != 1 or len(selected_wildrefer) != 1):
            raise ValueError(
                'strefer/liferefer currently support standalone training only '
                '(single dataset, no mixed training).'
            )
        if args.joint_det and not selected_wildrefer:
            dataset_dict['scannet'] = 10
        elif args.joint_det and selected_wildrefer:
            print(f'Warning: --joint_det is ignored for {selected_wildrefer[0]} dataset.')
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
                wildrefer_frame_num=args.wildrefer_frame_num,
                wildrefer_fuse_frames=args.wildrefer_fuse_frames,
                wildrefer_use_proj_geometry=args.wildrefer_use_proj_geometry,
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
            wildrefer_frame_num=args.wildrefer_frame_num,
            wildrefer_fuse_frames=args.wildrefer_fuse_frames,
            wildrefer_use_proj_geometry=args.wildrefer_use_proj_geometry,
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
        if args.wildrefer_use_proj_geometry:
            num_input_channel += 4
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
            external_attn_k=dict(
                bi_layer0=args.external_attn_k_keep0 or None,
                bi_layer1=args.external_attn_k_keep1 or None,
                com_trans=args.external_attn_k_com or None,
                seg_128=args.external_attn_k_seg128 or None,
                seg_64=args.external_attn_k_seg64 or None,
            ),
            mink_conv1_stride=args.mink_conv1_stride,
            top_pts_threshold=args.top_pts_threshold,
            top_pts_threshold_det=args.top_pts_threshold_det,
            prune_threshold=(args.prune_threshold_0, args.prune_threshold_1),
            pts_prune_threshold=tuple(args.pts_prune_threshold),
            random_prune_threshold=tuple(args.random_prune_threshold),
            test_cfg=dict(
                nms_pre=args.nms_pre,
                iou_thr=args.nms_iou_thr,
                score_thr=args.nms_score_thr,
            ),
            measure_fps_detail=args.measure_fps_detail
        )
        return model

    # BRIEF input data.
    @staticmethod
    def _get_inputs(batch_data):
        # print(batch_data['utterances'])
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(), # ([B, 50000, 6]) xyz + colour
            'text': batch_data['utterances'],                   # list[B]  text
            'target_cat': batch_data['target_cat']
        }
        if 'point_valid_mask' in batch_data:
            inputs['point_valid_mask'] = batch_data['point_valid_mask']
        return inputs


    @staticmethod
    @torch.no_grad()
    def evaluate_grounding(test_loader, model, args):
        """
        Core grounding eval loop. Returns a metrics dict with accuracy,
        timing, and memory stats. No logging side-effects.

        Keys always present:
            acc0.25, acc0.50, elapsed_s,
            avg_inf_s, avg_latency_ms, fps (fps=0 when measure_fps disabled),
            ext_bi0_ms, ext_bi1_ms, ext_bi2_ms, ext_total_ms,
            mem_avg_alloc_mib, mem_avg_res_mib,
            mem_peak_alloc_mib, mem_peak_res_mib,
            timing_visual_s, timing_text_s, timing_fusion_s, timing_head_s,
            stage_profile (dict, populated when measure_fps_detail=True)
        """
        import time as _time

        model.eval()
        evaluator = GroundingEvaluator(
            only_root=True, thresholds=[0.25, 0.5],
            topks=[1], prefixes=['3dcnn'],
            filter_non_gt_boxes=args.butd_cls,
            use_seg=args.use_seg,
        )

        fps_enabled      = bool(getattr(args, 'measure_fps', False))
        fps_warmup_iters = max(int(getattr(args, 'fps_warmup_iters', 20)), 0)
        fps_max_iters    = int(getattr(args, 'fps_max_iters', -1))

        inf_speeds, vis_speeds, text_speeds, fusion_speeds, head_speeds = [], [], [], [], []
        ext_bi0, ext_bi1, ext_bi2, ext_total = [], [], [], []
        stage_profile_sums, stage_profile_counts = {}, {}
        total_fps_samples = 0
        total_fps_time    = 0.0
        total_ext_time    = 0.0

        mem_allocated_mb, mem_reserved_mb = [], []
        max_allocated_mb = max_reserved_mb = 0.0
        collect_wildrefer_official = args.test_dataset in {'strefer', 'liferefer'}
        wildrefer_pred_boxes = []
        wildrefer_gt_boxes = []
        mem_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if mem_device is not None:
            torch.cuda.reset_peak_memory_stats(mem_device)

        stat_dict = {}
        t_wall0 = _time.time()

        pbar = tqdm(test_loader, ncols=TQDM_NCOLS, leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            if fps_max_iters > 0 and batch_idx >= fps_max_iters:
                break

            stat_dict, end_points, inf_speed, backbone_time, detail_time = \
                TrainTester._main_eval_branch(
                    batch_idx, batch_data, pbar, model,
                    stat_dict, None, None, args
                )

            inf_speeds.append(inf_speed)
            vis_speeds.append(detail_time[0])
            text_speeds.append(detail_time[1])
            fusion_speeds.append(detail_time[2])
            head_speeds.append(detail_time[3])

            head_module = model.module.head if hasattr(model, 'module') else model.head
            ext_profile = getattr(head_module, 'last_external_attn_profile', None) or \
                          {'bi_layer0': 0.0, 'bi_layer1': 0.0, 'bi_layer2': 0.0, 'total': 0.0}
            ext_bi0.append(float(ext_profile.get('bi_layer0', 0.0)))
            ext_bi1.append(float(ext_profile.get('bi_layer1', 0.0)))
            ext_bi2.append(float(ext_profile.get('bi_layer2', 0.0)))
            ext_total.append(float(ext_profile.get('total', 0.0)))

            if getattr(args, 'measure_fps_detail', False):
                stage_profile = getattr(head_module, 'last_stage_profile', None)
                if stage_profile is not None:
                    for key, value in stage_profile.items():
                        stage_profile_sums[key]   = stage_profile_sums.get(key, 0.0)  + float(value)
                        stage_profile_counts[key] = stage_profile_counts.get(key, 0) + 1

            if fps_enabled and batch_idx >= fps_warmup_iters:
                batch_size = int(batch_data['point_clouds'].shape[0])
                total_fps_samples += batch_size
                total_fps_time    += float(inf_speed)
                total_ext_time    += float(ext_profile.get('total', 0.0))

            if mem_device is not None and batch_idx >= fps_warmup_iters:
                mem_allocated_mb.append(torch.cuda.memory_allocated(mem_device) / (1024.0 ** 2))
                mem_reserved_mb.append(torch.cuda.memory_reserved(mem_device)   / (1024.0 ** 2))
                max_allocated_mb = max(max_allocated_mb,
                                       torch.cuda.max_memory_allocated(mem_device) / (1024.0 ** 2))
                max_reserved_mb  = max(max_reserved_mb,
                                       torch.cuda.max_memory_reserved(mem_device)  / (1024.0 ** 2))

            if collect_wildrefer_official:
                for bid in range(len(end_points['bbox_results'])):
                    scores = end_points['bbox_results'][bid]['scores_3d']
                    bboxes = end_points['bbox_results'][bid]['bboxes_3d']
                    bboxes = torch.cat([bboxes.gravity_center, bboxes.dims], dim=1)
                    pred_box = np.zeros((7,), dtype=np.float32)
                    box_scores = scores_to_box_scores(scores, bboxes.shape[0])
                    if box_scores.numel() > 0:
                        best_idx = int(torch.argmax(box_scores).item())
                        pred_box[:6] = bboxes[best_idx].detach().cpu().numpy().astype(np.float32)
                    gt_box = end_points['wildrefer_bbox7'][bid].detach().cpu().numpy().astype(np.float32)
                    wildrefer_pred_boxes.append(pred_box)
                    wildrefer_gt_boxes.append(gt_box)

            evaluator.evaluate(end_points, '3dcnn')

        elapsed = _time.time() - t_wall0
        evaluator.synchronize_between_processes()

        dets = evaluator.dets
        gts  = evaluator.gts
        acc25 = dets[('3dcnn', 0.25, 1, 'bbf')] / max(gts[('3dcnn', 0.25, 1, 'bbf')], 1)
        acc50 = dets[('3dcnn', 0.50, 1, 'bbf')] / max(gts[('3dcnn', 0.50, 1, 'bbf')], 1)

        avg_inf = float(np.mean(inf_speeds)) if inf_speeds else 0.0
        fps_val = (total_fps_samples / total_fps_time) if (fps_enabled and total_fps_time > 0) else 0.0

        stage_profile_means = {
            k: stage_profile_sums[k] / max(stage_profile_counts.get(k, 1), 1)
            for k in stage_profile_sums
        }

        if collect_wildrefer_official:
            if dist.is_initialized():
                gathered_pred = [None for _ in range(dist.get_world_size())]
                gathered_gt = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_pred, wildrefer_pred_boxes)
                dist.all_gather_object(gathered_gt, wildrefer_gt_boxes)
                merged_pred = [item for rank_list in gathered_pred for item in rank_list]
                merged_gt = [item for rank_list in gathered_gt for item in rank_list]
            else:
                merged_pred = wildrefer_pred_boxes
                merged_gt = wildrefer_gt_boxes
            official_acc25, official_acc50, official_miou = wildrefer_cal_accuracy(merged_pred, merged_gt)
        else:
            official_acc25, official_acc50, official_miou = 0.0, 0.0, 0.0

        metrics = {
            'acc0.25':            float(acc25),
            'acc0.50':            float(acc50),
            'official_acc0.25':   float(official_acc25),
            'official_acc0.50':   float(official_acc50),
            'official_miou':      float(official_miou),
            'elapsed_s':          round(elapsed, 3),
            'avg_inf_s':          round(avg_inf, 6),
            'avg_latency_ms':     round(avg_inf * 1000, 3),
            'fps':                round(fps_val, 3),
            'ext_bi0_ms':         round(float(np.mean(ext_bi0))   * 1000, 3) if ext_bi0   else 0.0,
            'ext_bi1_ms':         round(float(np.mean(ext_bi1))   * 1000, 3) if ext_bi1   else 0.0,
            'ext_bi2_ms':         round(float(np.mean(ext_bi2))   * 1000, 3) if ext_bi2   else 0.0,
            'ext_total_ms':       round(float(np.mean(ext_total)) * 1000, 3) if ext_total else 0.0,
            'mem_avg_alloc_mib':  round(float(np.mean(mem_allocated_mb)), 2) if mem_allocated_mb else 0.0,
            'mem_avg_res_mib':    round(float(np.mean(mem_reserved_mb)),  2) if mem_reserved_mb  else 0.0,
            'mem_peak_alloc_mib': round(max_allocated_mb, 2),
            'mem_peak_res_mib':   round(max_reserved_mb,  2),
            'timing_visual_s':    round(float(np.mean(vis_speeds)),    6) if vis_speeds    else 0.0,
            'timing_text_s':      round(float(np.mean(text_speeds)),   6) if text_speeds   else 0.0,
            'timing_fusion_s':    round(float(np.mean(fusion_speeds)), 6) if fusion_speeds else 0.0,
            'timing_head_s':      round(float(np.mean(head_speeds)),   6) if head_speeds   else 0.0,
            'stage_profile':      stage_profile_means,
            # pass through for evaluate_one_epoch logging
            '_evaluator':         evaluator,
            '_fps_enabled':       fps_enabled,
            '_fps_warmup_iters':  fps_warmup_iters,
            '_total_fps_samples': total_fps_samples,
            '_total_fps_time':    total_fps_time,
            '_total_ext_time':    total_ext_time,
            '_mem_device_avail':  mem_device is not None,
            '_mem_measured':      len(mem_allocated_mb) > 0,
        }
        if args.use_seg:
            metrics['acc_mask0.25'] = evaluator.dets['overall_mask']  / max(evaluator.gts['mask_3dcnn'], 1e-14)
            metrics['acc_mask0.50'] = evaluator.dets['overall50_mask'] / max(evaluator.gts['mask_3dcnn'], 1e-14)
        return metrics

    # BRIEF only eval one epoch.
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """Eval grounding after a single epoch."""
        if args.test_dataset == 'scannet':
            return self.evaluate_one_epoch_det(
                epoch, test_loader, model, criterion, set_criterion, args
            )

        m = TrainTester.evaluate_grounding(test_loader, model, args)

        if dist.get_rank() != 0:
            return None

        evaluator        = m['_evaluator']
        fps_enabled      = m['_fps_enabled']
        fps_warmup_iters = m['_fps_warmup_iters']

        evaluator.print_stats()
        self.logger.info(f'Eval: [{epoch}]  ')
        for t in evaluator.thresholds:
            self.logger.info(
                f"3dcnn Acc{t:.2f}: Top-1: "
                f"{evaluator.dets[('3dcnn', t, 1, 'bbf')] / max(evaluator.gts[('3dcnn', t, 1, 'bbf')], 1):.5f}"
            )
        if args.test_dataset in {'strefer', 'liferefer'}:
            self.logger.info(
                "WildRefer-official Acc0.25: %.5f | Acc0.50: %.5f | mIoU: %.5f"
                % (m['official_acc0.25'], m['official_acc0.50'], m['official_miou'])
            )
        if args.use_seg:
            self.logger.info('Acc_mask0.25 ' + str(evaluator.dets['overall_mask']   / evaluator.gts['mask_3dcnn']))
            self.logger.info('Acc_mask0.50 ' + str(evaluator.dets['overall50_mask'] / evaluator.gts['mask_3dcnn']))

        self.logger.info(
            'Timing(s): '
            f"inf={m['avg_inf_s']:.6f}, "
            f"visual={m['timing_visual_s']:.6f}, "
            f"text={m['timing_text_s']:.6f}, "
            f"fusion_wo_head={m['timing_fusion_s']:.6f}, "
            f"head={m['timing_head_s']:.6f}"
        )
        self.logger.info(
            'External-attn-affected module time(s): '
            f"bi_layer0={m['ext_bi0_ms']/1000:.6f}, "
            f"bi_layer1={m['ext_bi1_ms']/1000:.6f}, "
            f"bi_layer2={m['ext_bi2_ms']/1000:.6f}, "
            f"total={m['ext_total_ms']/1000:.6f}"
        )
        if m['stage_profile']:
            detail_items = [f"{k}={v:.6f}" for k, v in sorted(m['stage_profile'].items())]
            self.logger.info('Detailed stage time(s): ' + ', '.join(detail_items))
        if m['_mem_measured']:
            self.logger.info(
                'GPU memory(MiB): '
                f"avg_allocated={m['mem_avg_alloc_mib']:.2f}, "
                f"avg_reserved={m['mem_avg_res_mib']:.2f}, "
                f"peak_allocated={m['mem_peak_alloc_mib']:.2f}, "
                f"peak_reserved={m['mem_peak_res_mib']:.2f}, "
                f'warmup_iters={fps_warmup_iters}'
            )
        elif m['_mem_device_avail']:
            self.logger.info(
                'GPU memory(MiB): N/A (no measured iterations; reduce --fps_warmup_iters or increase eval iters).'
            )
        if fps_enabled:
            if m['_total_fps_samples'] > 0 and m['_total_fps_time'] > 0:
                self.logger.info(
                    f"FPS(single-card): {m['fps']:.3f} | Avg latency: {m['avg_latency_ms']:.3f} ms/sample "
                    f"| warmup_iters={fps_warmup_iters} | measured_samples={m['_total_fps_samples']}"
                )
                if m['_total_ext_time'] > 0:
                    ext_fps   = m['_total_fps_samples'] / m['_total_ext_time']
                    ext_ratio = m['_total_ext_time'] / m['_total_fps_time']
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

        test_loader = tqdm(test_loader, ncols=TQDM_NCOLS)
        for batch_idx, batch_data in enumerate(test_loader):
            # note eval
            stat_dict, end_points, _, _, _ = TrainTester._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args, logger=self.logger
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
    if opt.gpu_mem_limit_gb > 0:
        total_mem_gb = (
            torch.cuda.get_device_properties(opt.local_rank).total_memory
            / (1024 ** 3)
        )
        mem_fraction = min(opt.gpu_mem_limit_gb / total_mem_gb, 1.0)
        set_mem_fraction = getattr(torch.cuda, 'set_per_process_memory_fraction', None)
        if set_mem_fraction is None:
            if opt.local_rank == 0:
                print('Warning: torch.cuda.set_per_process_memory_fraction is unavailable in this PyTorch build.')
        else:
            set_mem_fraction(mem_fraction, device=opt.local_rank)
            if opt.local_rank == 0:
                print(
                    'GPU memory cap -> '
                    f'limit_gb={opt.gpu_mem_limit_gb}, '
                    f'total_gb={total_mem_gb:.2f}, '
                    f'mem_fraction={mem_fraction:.4f}'
                )
    # https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1304721237
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))  
    set_random_seed(opt.rng_seed + opt.local_rank)
    
    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = bool(opt.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(opt.cudnn_deterministic)

    if opt.use_deterministic_algorithms:
        # torch.use_deterministic_algorithms(True, warn_only=opt.deterministic_warn_only)
        torch.use_deterministic_algorithms(True)

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
