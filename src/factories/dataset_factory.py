# %%
import re
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from typing import Optional, Any, Dict, Tuple
from nptyping import NDArray


class WheatDataset(Dataset):
    def __init__(
        self,
        DIR_INPUT: str,
        mode: str = "train",
        transforms: Optional[A.Compose] = None,
    ):

        df: pd.DataFrame = self._read_data(DIR_INPUT, mode)

        self.image_ids: NDArray[np.object] = df["image_id"].unique()
        self.df: pd.DataFrame = df
        self.image_dir: str = f"{DIR_INPUT}/{mode}"
        self.transforms: Optional[A.Compose] = transforms

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[NDArray[(1024, 1024, 3), np.int], Dict[str, Any], str]:
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
            sample = {"image": image, "bboxes": target["bboxes"], "labels": labels}
            sample = self.transforms(**sample)
            image = sample["image"]
            target["bboxes"] = torch.stack(
                tuple(map(torch.tensor, zip(*sample["bboxes"])))
            ).permute(1, 0)
        return image, target, image_id

    @staticmethod
    def _read_data(DIR_INPUT: str, mode: str = "train") -> pd.DataFrame:
        def _expand_bbox(x):
            r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
            if len(r) == 0:
                r = [-1, -1, -1, -1]
            return r

        df = pd.read_csv(f"{DIR_INPUT}/{mode}.csv")
        for col in ["x", "y", "w", "h"]:
            df[col] = -1
        df[["x", "y", "w", "h"]] = np.stack(df["bbox"].apply(lambda x: _expand_bbox(x)))
        df.drop(columns=["bbox"], axis=1, inplace=True)
        for col in ["x", "y", "w", "h"]:
            df[col] = df[col].astype(np.float)
        df["x_max"] = df["x"] + df["w"]
        df["y_max"] = df["y"] + df["h"]

        df.rename({"x": "x_min", "y": "y_min"}, axis=1, inplace=True)
        return df

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
    DIR_INPUT: str, mode: str = "train", transforms: Optional[A.Compose] = None
):
    return WheatDataset(DIR_INPUT, mode=mode, transforms=transforms)
