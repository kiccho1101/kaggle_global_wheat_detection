import re
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset

import albumentations as A

from typing import Optional, Any, Dict, Tuple, List, Union
from nptyping import NDArray


class WheatDataset(Dataset):
    def __init__(
        self,
        INPUT_DIR: str,
        image_ids: NDArray[np.object],
        df: pd.DataFrame,
        mode: str = "train",
        transforms: Optional[A.Compose] = None,
    ):
        self.df: pd.DataFrame = df
        self.image_ids: NDArray[np.object] = image_ids
        self.mode: str = mode
        self.image_dir: str = f"{INPUT_DIR}/{mode}"
        self.transforms: Optional[A.Compose] = transforms

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, NDArray[(1024, 1024, 3), np.int], Dict[str, Any], str]:
        image_id: str = self.image_ids[index]
        records: pd.DataFrame = self.df[self.df["image_id"] == image_id]

        image: NDArray[(1024, 1024, 3), np.int] = self._read_image(image_id)
        bboxes: NDArray[(Any, 4), np.float] = records[
            ["x_min", "y_min", "x_max", "y_max"]
        ].values

        area: torch.Tensor = self._get_area(bboxes)
        labels: torch.Tensor = torch.ones((records.shape[0],), dtype=torch.int64)
        iscrowd: torch.Tensor = torch.zeros((records.shape[0],), dtype=torch.int64)

        target: Dict[str, Any] = {
            "bboxes": bboxes,
            "area": area,
            "labels": labels,
            "iscrowd": iscrowd,
            "mask": None,
        }

        if self.transforms:
            sample = self.transforms(
                **{"image": image, "bboxes": target["bboxes"], "labels": labels}
            )
            image = sample["image"]
            target["bboxes"] = torch.tensor(sample["bboxes"])
            target["bboxes"][:, [0, 1, 2, 3]] = target["bboxes"][:, [1, 0, 3, 2]]
            target["labels"] = torch.stack(sample["labels"])

        return image, target, image_id

    def _read_image(self, image_id: str) -> np.ndarray:
        image: np.ndarray = cv2.imread(
            f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image

    @staticmethod
    def _get_area(bboxes: NDArray[(Any, 4), np.float]) -> torch.Tensor:
        area: torch.Tensor = (bboxes[:, 3] - bboxes[:, 1]) * (
            bboxes[:, 2] - bboxes[:, 0]
        )
        area = torch.as_tensor(area, dtype=torch.float32)
        return area


def get_wheat_dataset(
    INPUT_DIR: str,
    df: pd.DataFrame,
    image_ids: NDArray[np.object],
    mode: str = "train",
    transforms: Optional[A.Compose] = None,
):
    return WheatDataset(INPUT_DIR, df, image_ids, mode=mode, transforms=transforms)
