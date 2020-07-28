import numpy as np
import torch
from src.types import Imgs
from src.factories.tta import TTACompose
from typing import List, Dict, Optional


def parse_det_result(
    det: torch.Tensor,
    images_len: int,
    score_threshold: float,
    tta_transform: Optional[TTACompose],
) -> List[Dict[str, np.ndarray]]:
    result: List[Dict[str, np.ndarray]] = []
    for i in range(images_len):
        bboxes = det[i].detach().cpu().numpy()[:, :4]
        scores = det[i].detach().cpu().numpy()[:, 4]
        indexes = np.where(scores > score_threshold)[0]
        bboxes = bboxes[indexes]
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
        if tta_transform is not None:
            bboxes = tta_transform.deaugment_boxes(bboxes.copy())
        result.append({"bboxes": bboxes, "scores": scores[indexes]})
    return result


def make_predictions(
    model: torch.nn.Module,
    images: Imgs,
    tta_transforms: Optional[List[TTACompose]],
    score_threshold: float = 0.25,
    device: torch.device = torch.device("cuda"),
) -> List[List[Dict[str, np.ndarray]]]:
    images = torch.stack(images).float().to(device)
    predictions: List[List[Dict[str, np.ndarray]]] = []

    if tta_transforms is not None:
        for tta_transform in tta_transforms:
            det = model(
                tta_transform.batch_augment(images.clone()),
                torch.tensor([1] * images.shape[0]).float().to(device),
            )
            predictions.append(
                parse_det_result(det, images.shape[0], score_threshold, tta_transform)
            )
    else:
        det = model(
            images.clone(), torch.tensor([1] * images.shape[0]).float().to(device),
        )
        predictions.append(
            parse_det_result(det, images.shape[0], score_threshold, None)
        )

    return predictions
