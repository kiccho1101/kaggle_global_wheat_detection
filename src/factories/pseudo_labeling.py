import numpy as np
import pandas as pd
from typing import Any, List


def get_pseudo_train_df(
    results: List[List[Any]], threshold: float = 0.05
) -> pd.DataFrame:
    for_df = []
    for image_id, bboxes, scores in results:
        bboxes = bboxes[scores >= float(threshold)]
        scores = scores[scores >= float(threshold)]
        bboxes = (bboxes * 2).astype(np.int32).clip(min=0, max=1023)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            w = x_max - x_min
            h = y_max - y_min

            for_df.append(
                {
                    "image_id": image_id,
                    "width": 1024,
                    "height": 1024,
                    "source": "",
                    "x_min": x_min,
                    "y_min": y_min,
                    "w": w,
                    "h": h,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            )
    return pd.DataFrame(for_df)
