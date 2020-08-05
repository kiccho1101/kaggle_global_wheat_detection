import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset
from src.types import Boxes

import albumentations as A

from src.types import Img
from typing import Optional, Any, Dict, Tuple, List, Union


class WheatDataset(Dataset):
    def __init__(
        self,
        INPUT_DIR: str,
        image_ids: np.ndarray,
        df: pd.DataFrame,
        mode: str = "train",
        transforms: Optional[A.Compose] = None,
        cutmix_ratio: float = 0.0,
    ):
        self.df: pd.DataFrame = df
        self.image_ids: np.ndarray = image_ids
        self.mode: str = mode
        self.INPUT_DIR = INPUT_DIR
        self.image_dir: str = f"{INPUT_DIR}/{mode}"
        self.transforms: Optional[A.Compose] = transforms
        self.cutmix_ratio: float = cutmix_ratio

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int) -> Union[torch.Tensor, Img, Dict[str, Any], str]:
        image_id: str = self.image_ids[index]

        image: Img = np.array([])
        bboxes: Boxes = np.array([])

        if self.mode == "test":
            image = self._read_image(image_id)
        else:
            if random.random() < self.cutmix_ratio:
                image, bboxes = self._load_cutmix_image_and_boxes(index)
            else:
                image = self._read_image(image_id)
                bboxes = self.df[self.df["image_id"] == image_id][
                    ["x_min", "y_min", "x_max", "y_max"]
                ].values

        labels: torch.Tensor = torch.ones((bboxes.shape[0],), dtype=torch.int64)

        target: Dict[str, Any] = {
            "bboxes": bboxes,
            "labels": labels,
        }

        if self.transforms:
            # Run the iteration until we find non-empty bbox
            if self.mode == "test":
                sample = self.transforms(**{"image": image})
                image = sample["image"]
            else:
                while True:
                    sample = self.transforms(
                        **{"image": image, "bboxes": target["bboxes"], "labels": labels}
                    )
                    if len(sample["bboxes"]) > 0:
                        image = sample["image"]
                        target["bboxes"] = torch.tensor(sample["bboxes"])
                        target["labels"] = torch.stack(sample["labels"])
                        target["bboxes"][:, [0, 1, 2, 3]] = target["bboxes"][
                            :, [1, 0, 3, 2]
                        ]  # to xyxy -> yxyx
                        break

        return image, target, image_id

    def _read_image(self, image_id: str) -> np.ndarray:
        if os.path.exists(f"{self.INPUT_DIR}/train/{image_id}.jpg"):
            image: np.ndarray = cv2.imread(
                f"{self.INPUT_DIR}/train/{image_id}.jpg", cv2.IMREAD_COLOR
            )
        else:
            image: np.ndarray = cv2.imread(
                f"{self.INPUT_DIR}/test/{image_id}.jpg", cv2.IMREAD_COLOR
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image

    def _load_cutmix_image_and_boxes(
        self, index: int, imsize: int = 1024,
    ):
        w, h = imsize, imsize
        s: int = imsize // 2

        xc: int = int(random.uniform(imsize * 0.25, imsize * 0.75))
        yc: int = int(random.uniform(imsize * 0.25, imsize * 0.75))

        indexes: List[int] = [index] + [
            random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)
        ]

        result_image: Img = np.full((imsize, imsize, 3), 1, dtype=np.float32)

        result_bboxes = []
        for i, idx in enumerate(indexes):
            image_id: str = self.image_ids[idx]
            image: Img = self._read_image(image_id)
            bboxes: Boxes = self.df[self.df["image_id"] == image_id][
                ["x_min", "y_min", "x_max", "y_max"]
            ].values

            x1a: int = 0
            y1a: int = 0
            x2a: int = 0
            y2a: int = 0
            x1b: int = 0
            y1b: int = 0
            x2b: int = 0
            y2b: int = 0
            if i == 0:
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)

                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (large image)

            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            bboxes[:, 0] += padw
            bboxes[:, 1] += padh
            bboxes[:, 2] += padw
            bboxes[:, 3] += padh

            result_bboxes.append(bboxes)

        result_bboxes = np.concatenate(result_bboxes, 0)
        np.clip(result_bboxes[:, 0:], 0, 2 * s, out=result_bboxes[:, 0:])
        result_bboxes = result_bboxes.astype(np.int32)
        result_bboxes = result_bboxes[
            np.where(
                (result_bboxes[:, 2] - result_bboxes[:, 0])
                * (result_bboxes[:, 3] - result_bboxes[:, 1])
                > 0
            )
        ]
        return result_image, result_bboxes


def get_wheat_dataset(
    INPUT_DIR: str,
    image_ids: np.ndarray,
    df: pd.DataFrame,
    mode: str = "train",
    transforms: Optional[A.Compose] = None,
    cutmix_ratio: float = 0.0,
):
    return WheatDataset(
        INPUT_DIR,
        image_ids,
        df,
        mode=mode,
        transforms=transforms,
        cutmix_ratio=cutmix_ratio,
    )
