import cv2
import matplotlib.pyplot as plt
import numpy as np
import numba
import torch
import torch.nn
from torch.utils.data import DataLoader

from src.config import Config
from src.types import Boxes, Imgs
from src.factories.model import get_effdet_eval
from src.factories.tta import TTACompose, get_tta_transforms
from src.factories.make_predictions import make_predictions
from src.factories.wbf import run_wbf
from src.factories.metric import calculate_image_precision

from tqdm.autonotebook import tqdm
from typing import List, Optional


def show_eval_result(
    checkpoint_path: str,
    config: Config,
    valid_loader: DataLoader,
    tta_transforms: Optional[List[TTACompose]],
    idx_start: int = 0,
    idx_end: int = 10,
):
    model = get_effdet_eval(checkpoint_path).to(config.device)
    image_index = 0
    for i, (images, targets, _) in tqdm(
        enumerate(valid_loader), total=len(valid_loader)
    ):
        if i >= idx_start and i <= idx_end:
            predictions = make_predictions(
                model, images, tta_transforms, 0.25, config.device
            )
            preds, scores, _ = run_wbf(
                predictions,
                image_index,
                image_size=512,
                iou_thr=0.44,
                skip_box_thr=0.43,
                weights=None,
            )
            preds_sorted_idx = np.argsort(scores)[::-1]
            preds_sorted = preds[preds_sorted_idx]
            iou_thresholds = numba.typed.List()
            for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                iou_thresholds.append(x)

            gts: Boxes = targets[image_index]["bboxes"].cpu().numpy()
            image_precision = calculate_image_precision(
                gts, preds_sorted, thresholds=iou_thresholds, form="pascal_voc"
            )

            print()
            print(f"No.{i} Score:{image_precision}")

            figsize = (7, 7)
            img = images[image_index]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
            for bbox in preds:
                img = cv2.rectangle(
                    img.copy(),
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (1, 0, 0),
                    2,
                )
            for bbox in gts:
                img = cv2.rectangle(
                    img.copy(),
                    (int(bbox[1]), int(bbox[0])),
                    (int(bbox[3]), int(bbox[2])),
                    (0, 0, 1),
                    2,
                )
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.show()
