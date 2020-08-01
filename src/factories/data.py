import re
import pandas as pd
import numpy as np

from src.config import Config

from sklearn.model_selection import StratifiedKFold


from typing import Tuple, List


class WheatData:
    def __init__(self, config: Config):
        self.config = config
        df: pd.DataFrame = self._read_df()

        self.image_ids: np.ndarray = df["image_id"].unique()
        self.df: pd.DataFrame = df
        self.df_folds: pd.DataFrame = self._get_df_folds(df)

    def get_fold(
        self, fold_num: int
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
        train_image_ids: np.ndarray = self.df_folds[self.df_folds["fold"] != fold_num][
            "image_id"
        ].values
        train_df: pd.DataFrame = self.df[self.df["image_id"].isin(train_image_ids)]
        train_df["folder"] = "train"

        val_image_ids: np.ndarray = self.df_folds[self.df_folds["fold"] == fold_num][
            "image_id"
        ].values
        val_df: pd.DataFrame = self.df[self.df["image_id"].isin(val_image_ids)]

        return train_image_ids, train_df, val_image_ids, val_df

    def _read_df(self) -> pd.DataFrame:
        def _expand_bbox(x):
            r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
            if len(r) == 0:
                r = [-1, -1, -1, -1]
            return r

        df = pd.read_csv(f"{self.config.INPUT_DIR}/train.csv")
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

    def _get_df_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        df_folds: pd.DataFrame = (
            df.groupby(["image_id", "source"])["image_id"]
            .count()
            .rename("bbox_count")
            .reset_index()
        )

        df_folds["stratify_group"] = df_folds.apply(
            lambda x: "{}_{}".format(x.source, x.bbox_count // 15), axis=1
        )

        df_folds["fold"] = 1

        if self.config.n_folds != 0:
            skf = StratifiedKFold(
                n_splits=self.config.n_folds, shuffle=True, random_state=42
            )
            for fold_num, (train_idx, val_idx) in enumerate(
                skf.split(X=df_folds.index, y=df_folds["stratify_group"])
            ):
                df_folds.loc[df_folds.iloc[val_idx].index, "fold"] = fold_num

        return df_folds


def get_data(config: Config) -> WheatData:
    return WheatData(config)
