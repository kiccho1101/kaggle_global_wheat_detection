from src.config import Config
from src.factories.make_predictions import make_predictions
from src.factories.wbf import run_wbf
from src.factories.tta import get_tta_transforms
from src.visualize import imshow_with_bboxes

import torch
from torch.utils.data import DataLoader
from typing import List, Any
from tqdm.autonotebook import tqdm


def get_results(
    test_loader: DataLoader, model: torch.nn.Module, config: Config
) -> List[List[Any]]:
    results: List[List[Any]] = []
    for images, targets, image_ids in tqdm(test_loader, total=len(test_loader)):
        for image_index, image in enumerate(images):
            predictions = make_predictions(
                model, images, get_tta_transforms(), 0.25, config.device
            )
            bboxes, scores, _ = run_wbf(predictions, image_index)
            imshow_with_bboxes(images[image_index], bboxes)
            results.append([image_ids[image_index], bboxes, scores])
    return results
