import pandas as pd
from typing import Any, List


def get_pseudo_train_df(
    results: List[List[Any]], threshold: float = 0.05
) -> pd.DataFrame:
    for_df = []
    for image_id, bboxes, scores in results:
        bboxes = bboxes[scores >= float(threshold)]
        scores = scores[scores >= float(threshold)]
        for bbox in bboxes:
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
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
                    "folder": "pseudo_test",
                }
            )
    return pd.DataFrame(for_df)
