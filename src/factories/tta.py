import numpy as np
import torch
import albumentations as A

from nptyping import NDArray
from typing import Any, List

from torch import result_type

Img = NDArray[(3, 512, 512), int]
Imgs = NDArray[(Any, 3, 512, 512), int]
Boxes = NDArray[(Any, 4), float]


class BaseWheatTTA:
    image_size = 512

    def augment(self, image: Img) -> Img:
        raise NotImplementedError

    def batch_augment(self, images: Imgs) -> Imgs:
        raise NotImplementedError

    def deaugment_boxes(self, boxes: Boxes) -> Boxes:
        raise NotImplementedError


class TTAHorizontalFlip(BaseWheatTTA):
    def augment(self, image: Img) -> Img:
        return image.flip(1)

    def batch_augment(self, images: Imgs) -> Imgs:
        return images.flip(2)

    def deaugment_boxes(self, boxes: Boxes) -> Boxes:
        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
        return boxes


class TTAVerticalFlip(BaseWheatTTA):
    def augment(self, image: Img) -> Img:
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images: Imgs) -> Imgs:
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes: Boxes) -> Boxes:
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [1, 3]]
        res_boxes[:, [1, 3]] = boxes[:, [2, 0]]
        return res_boxes


class TTACompose(BaseWheatTTA):
    def __init__(self, transforms: List[BaseWheatTTA]):
        self.transforms = transforms

    def augment(self, image: Img) -> Img:
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images: Img) -> Img:
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes: Boxes) -> Boxes:
        result_boxes: Boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes: Boxes) -> Boxes:
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)
