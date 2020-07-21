import ensemble_boxes
import numpy as np

from typing import List, Dict, Optional, Tuple, Any
from nptyping import NDArray
from src.types import Boxes


def run_wbf(
    predictions: List[List[Dict[str, np.ndarray]]],
    image_index: int,
    image_size: int = 512,
    iou_thr: float = 0.44,
    skip_box_thr: float = 0.43,
    weights: Optional[int] = None,
) -> Tuple[Boxes, NDArray[(Any), float], NDArray[(Any), float]]:
    bboxes: List[List[List[float]]] = [
        (prediction[image_index]["bboxes"] / (image_size - 1)).tolist()
        for prediction in predictions
    ]
    scores: List[List[float]] = [
        prediction[image_index]["scores"].tolist() for prediction in predictions
    ]
    labels: List[List[int]] = [
        np.ones(prediction[image_index]["scores"].shape[0]).astype(int).tolist()
        for prediction in predictions
    ]
    bboxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(
        bboxes,
        scores,
        labels,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    bboxes = bboxes * (image_size - 1)

    return bboxes, scores, labels
