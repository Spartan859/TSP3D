#!/usr/bin/env python3
"""Parity checks between official WildRefer data flow and TSP3D WildRefer path."""

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial import Delaunay
from transformers import RobertaTokenizerFast
import spacy


@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@dataclass
class CheckResult:
    index: int
    field: str
    expected: str
    got: str


def _resolve_default_paths(repo_root: str):
    candidates = [
        os.path.join(repo_root, "data", "WildRefer"),
        os.path.abspath(os.path.join(repo_root, "..", "WildRefer")),
    ]
    wildrefer_root = None
    for candidate in candidates:
        if os.path.exists(candidate):
            wildrefer_root = candidate
            break
    if wildrefer_root is None:
        wildrefer_root = candidates[0]
    data_root = os.path.join(repo_root, "data")
    return wildrefer_root, data_root


def _build_official_annos(wildrefer_root: str, dataset_name: str, split: str) -> List[Dict]:
    split_file = f"{dataset_name}_{'train' if split == 'train' else 'test'}.json"
    candidates = [
        os.path.join(wildrefer_root, split_file),
        os.path.join(wildrefer_root, "data", split_file),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"Official annotation json not found: {candidates}")


def _load_meta(wildrefer_root: str, dataset_name: str):
    src_name = "STRefer" if dataset_name == "strefer" else "LifeRefer"
    src_root = os.path.join(wildrefer_root, "src", src_name)
    prev_file = (
        "find_previous_strefer.json"
        if dataset_name == "strefer"
        else "find_previous_liferefer.json"
    )
    with open(os.path.join(src_root, prev_file)) as f:
        find_previous = json.load(f)
    points2image = None
    if dataset_name == "strefer":
        with open(os.path.join(src_root, "points2image_strefer.json")) as f:
            points2image = json.load(f)
    return src_root, find_previous, points2image


def _build_tsp3d_dataset(
    repo_root: str,
    dataset_name: str,
    split: str,
    data_root: str,
):
    sys.path.insert(0, repo_root)
    from src.joint_det_dataset import Joint3DDataset

    with pushd(repo_root):
        return Joint3DDataset(
            dataset_dict={dataset_name: 1},
            test_dataset=dataset_name,
            split=split,
            data_path=data_root if data_root.endswith("/") else data_root + "/",
            use_color=False,
            use_height=False,
            use_multiview=False,
            detect_intermediate=False,
            butd=False,
            butd_gt=False,
            butd_cls=False,
            augment_det=False,
        )


def _random_sampling(pc: np.ndarray, num_sample: int = 30000) -> np.ndarray:
    replace = pc.shape[0] < num_sample
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    return pc[choices]


def _rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _my_compute_box_3d(center, size, heading_angle):
    rot = _rotz(-1 * heading_angle)
    l, w, h = size
    l /= 2
    w /= 2
    h /= 2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(rot, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def _extract_pc_in_box3d(pc: np.ndarray, box3d: np.ndarray) -> np.ndarray:
    hull = Delaunay(box3d)
    inds = hull.find_simplex(pc[:, 0:3]) >= 0
    return inds


def _get_positive_map(tokenized, tokens_positive, max_lang_num):
    positive_map = torch.zeros((len(tokens_positive), max_lang_num), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        beg, end = int(tok_list[0]), int(tok_list[1])
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except Exception:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except Exception:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos : end_pos + 1].fill_(1)
    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()


def _official_token_positive_map(
    description: str,
    max_obj_num: int,
    max_lang_num: int,
    nlp,
    tokenizer,
) -> Tuple[np.ndarray, np.ndarray]:
    caption = " ".join(description.replace(",", " ,").split())
    caption = " " + caption + " "
    tokens_positive = np.zeros((max_obj_num, 2))
    doc = nlp(caption)
    cat_names = []
    for token in doc:
        if token.dep_ == "nsubj":
            cat_names.append(token.text)
            break
    if len(cat_names) <= 0:
        for token in doc:
            if token.dep_ == "ROOT":
                cat_names.append(token.text)
                break

    for c, cat_name in enumerate(cat_names):
        start_span = caption.find(" " + cat_name + " ")
        span_len = len(cat_name)
        if start_span < 0:
            start_span = caption.find(" " + cat_name)
            span_len = len(caption[start_span + 1 :].split()[0])
        if start_span < 0:
            start_span = caption.find(cat_name)
            orig_start = start_span
            while caption[start_span - 1] != " ":
                start_span -= 1
            span_len = len(cat_name) + orig_start - start_span
            while caption[span_len + start_span] != " ":
                span_len += 1
        end_span = start_span + span_len
        tokens_positive[c][0] = start_span
        tokens_positive[c][1] = end_span

    tokenized = tokenizer.batch_encode_plus(
        [" ".join(description.replace(",", " ,").split())],
        padding="longest",
        return_tensors="pt",
    )
    positive_map = np.zeros((max_obj_num, max_lang_num))
    gt_map = _get_positive_map(tokenized, tokens_positive[: len(cat_names)], max_lang_num)
    positive_map[: len(cat_names)] = gt_map
    return tokens_positive, positive_map


def _official_expected(
    anno: Dict,
    dataset_name: str,
    src_root: str,
    find_previous: Dict,
    points2image: Dict,
    frame_num: int,
    max_obj_num: int,
    max_lang_num: int,
    nlp,
    tokenizer,
) -> Dict:
    scene_id = str(anno["scene_id"])
    point_cloud_name = str(anno["point_cloud"]["point_cloud_name"])
    target_bbox = np.array(anno["point_cloud"]["bbox"], dtype=np.float32)
    description = str(anno["language"]["description"]).lower()

    scene_file = os.path.join(src_root, "points_rgbd", scene_id, f"{point_cloud_name}.npy")
    scene = np.load(scene_file).astype(np.float32)
    scene[:, 3:6] = scene[:, 3:6] / 255.0
    scene = _random_sampling(scene, 30000)

    dynamic_mask = [1]
    prev_name = point_cloud_name
    for _ in range(1, frame_num):
        if prev_name:
            prev_name = find_previous[scene_id][prev_name]
            if dataset_name == "strefer" and prev_name:
                _ = points2image[scene_id][prev_name]
        if prev_name:
            dynamic_mask.append(1)
        else:
            dynamic_mask.append(0)

    gt_boxes3d = np.zeros((max_obj_num, 6), dtype=np.float32)
    gt_boxes3d[0] = target_bbox[:6]
    point_instance_label = -np.ones(len(scene), dtype=np.int64)
    instance_ind = _extract_pc_in_box3d(
        scene.copy(), _my_compute_box_3d(target_bbox[0:3], target_bbox[3:6], target_bbox[6])
    )
    point_instance_label[instance_ind] = 0

    tokens_positive, positive_map = _official_token_positive_map(
        description=description,
        max_obj_num=max_obj_num,
        max_lang_num=max_lang_num,
        nlp=nlp,
        tokenizer=tokenizer,
    )

    text = " ".join(description.replace(",", " ,").replace(".", " .").split()) + " not mentioned"

    return {
        "dynamic_mask": np.asarray(dynamic_mask, dtype=np.int64),
        "center_label": gt_boxes3d[:, :3],
        "size_gts": gt_boxes3d[:, 3:6],
        "point_instance_label": point_instance_label,
        "tokens_positive": tokens_positive.astype(np.int64),
        "positive_map": positive_map.astype(np.float32),
        "text": text,
        "point_count": int(scene.shape[0]),
    }


def _fmt(arr) -> str:
    return np.array2string(np.asarray(arr), precision=4, separator=",")


def _check_one(index: int, official: Dict, adapted: Dict) -> List[CheckResult]:
    mismatches: List[CheckResult] = []

    checks = [
        (
            "dynamic_mask",
            np.asarray(official["dynamic_mask"]).tolist(),
            np.asarray(adapted["wildrefer_dynamic_mask"]).tolist(),
        ),
        (
            "center_label[0]",
            np.asarray(official["center_label"][0]),
            np.asarray(adapted["center_label"][0]),
        ),
        (
            "size_gts[0]",
            np.asarray(official["size_gts"][0]),
            np.asarray(adapted["size_gts"][0]),
        ),
        ("text", str(official["text"]), str(adapted["utterances"])),
        (
            "tokens_positive[0]",
            np.asarray(official["tokens_positive"][0]),
            np.asarray(adapted["tokens_positive"][0]),
        ),
        (
            "positive_map_nonzero",
            np.nonzero(np.asarray(official["positive_map"][0]))[0],
            np.nonzero(np.asarray(adapted["positive_map"][0]))[0],
        ),
        (
            "point_instance_foreground_count",
            int((np.asarray(official["point_instance_label"]) == 0).sum()),
            int((np.asarray(adapted["point_instance_label"]) == 0).sum()),
        ),
        (
            "point_count",
            int(official["point_count"]),
            int(np.asarray(adapted["point_clouds"]).shape[0]),
        ),
    ]

    for field, expected, got in checks:
        if isinstance(expected, str):
            same = expected == got
        elif np.isscalar(expected):
            same = expected == got
        else:
            same = np.array_equal(np.asarray(expected), np.asarray(got))
        if not same:
            mismatches.append(
                CheckResult(index=index, field=field, expected=_fmt(expected), got=_fmt(got))
            )
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Check WildRefer loader parity with official code.")
    parser.add_argument("--dataset", choices=["strefer", "liferefer"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=3)
    parser.add_argument("--max-lang-num", type=int, default=256)
    parser.add_argument("--max-obj-num", type=int, default=132)
    parser.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--wildrefer-root", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    wildrefer_root_default, data_root_default = _resolve_default_paths(args.repo_root)
    wildrefer_root = args.wildrefer_root or wildrefer_root_default
    data_root = args.data_root or data_root_default

    annos = _build_official_annos(wildrefer_root=wildrefer_root, dataset_name=args.dataset, split=args.split)
    src_root, find_previous, points2image = _load_meta(
        wildrefer_root=wildrefer_root, dataset_name=args.dataset
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(
        os.path.join(data_root, "roberta-base"), local_files_only=True
    )
    nlp = spacy.load("en_core_web_sm")
    adapted_ds = _build_tsp3d_dataset(
        repo_root=args.repo_root,
        dataset_name=args.dataset,
        split=args.split,
        data_root=data_root,
    )

    if len(annos) != len(adapted_ds):
        print(
            f"[FAIL] dataset length mismatch: official={len(annos)} adapted={len(adapted_ds)}",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    sample_count = min(args.samples, len(annos))
    indices = sorted(rng.sample(range(len(annos)), sample_count))

    all_mismatches: List[CheckResult] = []
    for idx in indices:
        np.random.seed(args.seed + idx)
        official_item = _official_expected(
            anno=annos[idx],
            dataset_name=args.dataset,
            src_root=src_root,
            find_previous=find_previous,
            points2image=points2image,
            frame_num=args.frame_num,
            max_obj_num=args.max_obj_num,
            max_lang_num=args.max_lang_num,
            nlp=nlp,
            tokenizer=tokenizer,
        )

        np.random.seed(args.seed + idx)
        with pushd(args.repo_root):
            adapted_item = adapted_ds[idx]

        all_mismatches.extend(_check_one(idx, official_item, adapted_item))

    if all_mismatches:
        print(f"[FAIL] {len(all_mismatches)} mismatches found across {len(indices)} samples.")
        for item in all_mismatches:
            print(
                f"  - idx={item.index} field={item.field}\n"
                f"    expected: {item.expected}\n"
                f"    got     : {item.got}"
            )
        return 1

    print(
        f"[OK] WildRefer parity checks passed for dataset={args.dataset}, split={args.split}, "
        f"samples={len(indices)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
