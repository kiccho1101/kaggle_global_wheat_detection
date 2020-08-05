import torch
import torch.nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from src.types import Boxes, Img
from src.factories.make_predictions import make_predictions
from src.factories.tta import TTACompose
from src.factories.wbf import run_wbf
from src.config import Config

from tqdm.autonotebook import tqdm
from typing import List, Dict, Any


def show_result(image: Img, bboxes: Boxes):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    for bbox in bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (1, 0, 0), 1)
    ax.set_axis_off()
    ax.imshow(image)
    plt.show()


def format_prediction_string(bboxes: Boxes, scores: np.ndarray):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return " ".join(
        [
            "{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3])
            for j in zip(scores, bboxes)
        ]
    )


def inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    tta_transforms: List[TTACompose],
    config: Config,
) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    for _, (images, _, image_ids) in tqdm(
        enumerate(test_loader), total=len(test_loader)
    ):
        predictions = make_predictions(
            model, images, tta_transforms, 0.25, config.device
        )
        for image_index in range(len(images)):
            bboxes, scores, _ = run_wbf(
                predictions,
                image_index,
                image_size=512,
                iou_thr=0.44,
                skip_box_thr=0.43,
                weights=None,
            )
            image = images[image_index].permute(1, 2, 0).cpu().numpy()
            bboxes = bboxes.astype(np.int32).clip(min=0, max=511)

            show_result(image, bboxes)

            bboxes = (bboxes * 2).astype(np.int32).clip(min=0, max=1023)
            results.append(
                {
                    "image_id": image_ids[image_index],
                    "PredictionString": format_prediction_string(bboxes, scores),
                }
            )
    return pd.DataFrame(results, columns=["image_id", "PredictionString"])
