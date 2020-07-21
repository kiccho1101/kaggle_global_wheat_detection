import numpy as np
import torch
import albumentations as A
from itertools import product

from src.types import Img, Imgs, Boxes
from typing import Any, List, Dict


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


class TTARotate90(BaseWheatTTA):
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


def get_tta_transforms() -> List[TTACompose]:
    tta_transforms: List[TTACompose] = [
        TTACompose(
            [tta_transform for tta_transform in tta_combination if tta_transform]
        )
        for tta_combination in product(
            [TTAHorizontalFlip(), None],
            [TTAVerticalFlip(), None],
            [TTARotate90(), None],
        )
    ]
    return tta_transforms


def make_tta_predictions(
    model: torch.nn.Module,
    images: Imgs,
    tta_transforms: List[TTACompose],
    score_threshold: float = 0.25,
    device: str = "cuda",
) -> List[List[Dict[str, np.ndarray]]]:
    images = torch.stack(images).float().to(device)
    predictions: List[List[Dict[str, np.ndarray]]] = []

    for tta_transform in tta_transforms:
        tta_transform = tta_transforms[0]
        det = model(
            tta_transform.batch_augment(images.clone()),
            torch.tensor([1] * images.shape[0]).float().to(device),
        )
        result: List[Dict[str, np.ndarray]] = []
        i = 0
        for i in range(images.shape[0]):
            bboxes = det[i].detach().cpu().numpy()[:, :4]
            scores = det[i].detach().cpu().numpy()[:, 4]
            indexes = np.where(scores > score_threshold)[0]
            bboxes = bboxes[indexes]
            bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
            bboxes = tta_transform.deaugment_boxes(bboxes.copy())
            result.append({"bboxes": bboxes, "scores": scores[indexes]})
        predictions.append(result)

    return predictions
