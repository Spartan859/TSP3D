import math

import numpy as np
from shapely.geometry import Polygon


def _cal_corner_after_rotation(corner, center, rotation):
    x1, y1 = corner
    x0, y0 = center
    x2 = math.cos(rotation) * (x1 - x0) - math.sin(rotation) * (y1 - y0) + x0
    y2 = math.sin(rotation) * (x1 - x0) + math.cos(rotation) * (y1 - y0) + y0
    return x2, y2


def _eight_points(center, size, rotation=0.0):
    x, y, z = center
    w, l, h = size
    w = w / 2
    l = l / 2
    h = h / 2

    x1, y1, z1 = x - w, y - l, z + h
    x2, y2, z2 = x + w, y - l, z + h
    x3, y3, z3 = x + w, y - l, z - h
    x4, y4, z4 = x - w, y - l, z - h
    x5, y5, z5 = x - w, y + l, z + h
    x6, y6, z6 = x + w, y + l, z + h
    x7, y7, z7 = x + w, y + l, z - h
    x8, y8, z8 = x - w, y + l, z - h

    if rotation != 0:
        x1, y1 = _cal_corner_after_rotation((x1, y1), (x, y), rotation)
        x2, y2 = _cal_corner_after_rotation((x2, y2), (x, y), rotation)
        x3, y3 = _cal_corner_after_rotation((x3, y3), (x, y), rotation)
        x4, y4 = _cal_corner_after_rotation((x4, y4), (x, y), rotation)
        x5, y5 = _cal_corner_after_rotation((x5, y5), (x, y), rotation)
        x6, y6 = _cal_corner_after_rotation((x6, y6), (x, y), rotation)
        x7, y7 = _cal_corner_after_rotation((x7, y7), (x, y), rotation)
        x8, y8 = _cal_corner_after_rotation((x8, y8), (x, y), rotation)

    corner1 = np.array([x1, y1, z1])
    corner2 = np.array([x2, y2, z2])
    corner3 = np.array([x3, y3, z3])
    corner4 = np.array([x4, y4, z4])
    corner5 = np.array([x5, y5, z5])
    corner6 = np.array([x6, y6, z6])
    corner7 = np.array([x7, y7, z7])
    corner8 = np.array([x8, y8, z8])

    return np.stack(
        [corner1, corner2, corner6, corner5, corner4, corner3, corner7, corner8],
        axis=0,
    )


def _cal_inter_area(box1, box2):
    a = np.array(box1).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    b = np.array(box2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0.0
    else:
        inter_area = poly1.intersection(poly2).area
    return poly1.area, poly2.area, inter_area


def cal_iou3d(box1, box2):
    center1 = box1[:3]
    size1 = box1[3:6]
    rotation1 = box1[6]
    corners1 = _eight_points(center1, size1, rotation1)

    center2 = box2[:3]
    size2 = box2[3:6]
    rotation2 = box2[6]
    corners2 = _eight_points(center2, size2, rotation2)

    area1, area2, inter_area = _cal_inter_area(
        corners1[:4, :2].reshape(-1), corners2[:4, :2].reshape(-1)
    )

    h1, z1 = box1[5], box1[2]
    h2, z2 = box2[5], box2[2]
    volume1 = h1 * area1
    volume2 = h2 * area2

    bottom1, top1 = z1 - h1 / 2, z1 + h1 / 2
    bottom2, top2 = z2 - h2 / 2, z2 + h2 / 2
    inter_bottom = max(bottom1, bottom2)
    inter_top = min(top1, top2)
    inter_h = inter_top - inter_bottom if inter_top > inter_bottom else 0.0

    inter_volume = inter_area * inter_h
    union_volume = volume1 + volume2 - inter_volume
    if union_volume <= 0:
        return 0.0
    return inter_volume / union_volume


def cal_accuracy(pred_bboxes, gt_bboxes):
    if len(pred_bboxes) != len(gt_bboxes):
        raise ValueError(
            f'pred_bboxes and gt_bboxes length mismatch: {len(pred_bboxes)} vs {len(gt_bboxes)}'
        )
    total = len(gt_bboxes)
    if total == 0:
        return 0.0, 0.0, 0.0
    tp25 = 0
    tp50 = 0
    miou = 0.0
    for i in range(total):
        iou = cal_iou3d(pred_bboxes[i][:7], gt_bboxes[i][:7])
        if iou >= 0.5:
            tp25 += 1
            tp50 += 1
        elif iou >= 0.25:
            tp25 += 1
        miou += iou

    return round(tp25 / total, 4), round(tp50 / total, 4), round(miou / total, 4)
