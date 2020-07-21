import numpy as np
import numba
from numba import jit

from typing import Optional, List
from src.types import Box, Boxes


@jit(nopython=True)
def calculate_iou(gt: Box, pr: Box, form: str = "pascal_voc") -> float:
    if form == "coco":
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx: float = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    dy: float = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    overlap_area: float = dx * dy

    if dx < 0 or dy < 0:
        return 0.0

    # Calculate union area
    union_area: float = (
        (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
        + (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1)
        - overlap_area
    )

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(
    gts: Boxes,
    pred: Box,
    pred_idx: numba.int32,
    threshold: float = 0.5,
    form: str = "pascal_voc",
    ious: Optional[np.ndarray] = None,
) -> numba.int32:
    best_match_iou: numba.float64 = -np.inf
    best_match_idx: numba.int32 = -1

    for gt_idx in range(len(gts)):
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou: numba.float64 = -1 if ious is None else ious[gt_idx][pred_idx]
        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def calculate_precision(
    gts: Boxes,
    preds: Boxes,
    threshold: numba.float64 = 0.5,
    form="pascal_voc",
    ious: Optional[np.ndarray] = None,
) -> float:
    tp: numba.int32 = 0
    fp: numba.int32 = 0

    for pred_idx in range(len(preds)):
        best_match_gt_idx = find_best_match(
            gts, preds[pred_idx], pred_idx, threshold=threshold, form=form, ious=ious
        )
        if best_match_gt_idx >= 0:
            tp += 1
            gts[best_match_gt_idx] = -1
        else:
            fp += 1
    fn = (gts.sum(axis=1) > 0).sum()
    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(
    gts: Boxes,
    preds: Boxes,
    thresholds: List[numba.float64] = [0.5, 0.76, 0.05],
    form: str = "pascal_voc",
) -> float:
    n_thresholds: int = int(len(thresholds))
    image_precision: float = float(0.0)
    ious = np.ones((len(gts), len(preds))) * -1

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(
            gts.copy(), preds, threshold=threshold, form=form, ious=ious
        )
        image_precision += precision_at_threshold / n_thresholds
    return image_precision
