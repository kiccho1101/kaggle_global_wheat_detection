import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import Tuple, Any, Union
from nptyping import NDArray

from .factories import WheatDataset


def imshow_with_bboxes(
    img: NDArray[np.int],
    bboxes: NDArray[(Any, 4), np.float],
    figsize: Tuple[int, int] = (7, 7),
):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()

    for bbox in bboxes:
        img = cv2.rectangle(
            img.copy(),
            # (int(bbox[1]), int(bbox[0])),
            # (int(bbox[3]), int(bbox[2])),
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (1, 0, 0),
            2,
        )
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
