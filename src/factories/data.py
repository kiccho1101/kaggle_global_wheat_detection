import re
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from nptyping import NDArray

from typing import Tuple


class WheatData:
    def __init__(self, INPUT_DIR: str):
        df: pd.DataFrame = self._read_df(INPUT_DIR)

        self.image_ids: NDArray[np.object] = df["image_id"].unique()
        self.df: pd.DataFrame = df
        self.df_folds: pd.DataFrame = self._get_df_folds(df)

    def get_fold(
        self, fold_num: int
    ) -> Tuple[NDArray[np.object], pd.DataFrame, NDArray[np.object], pd.DataFrame]:
        train_image_ids: NDArray[np.object] = self.df_folds[
            self.df_folds["fold"] != fold_num
        ]["image_id"].values
        train_df: pd.DataFrame = self.df[self.df["image_id"].isin(train_image_ids)]

        val_image_ids: NDArray[np.object] = self.df_folds[
            self.df_folds["fold"] == fold_num
        ]["image_id"].values
        val_df: pd.DataFrame = self.df[self.df["image_id"].isin(val_image_ids)]

        return train_image_ids, train_df, val_image_ids, val_df

    @staticmethod
    def _read_df(INPUT_DIR: str) -> pd.DataFrame:
        def _expand_bbox(x):
            r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
            if len(r) == 0:
                r = [-1, -1, -1, -1]
            return r

        df = pd.read_csv(f"{INPUT_DIR}/train.csv")
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

    @staticmethod
    def _get_df_folds(df: pd.DataFrame) -> pd.DataFrame:
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
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_num, (train_idx, val_idx) in enumerate(
            skf.split(X=df_folds.index, y=df_folds["stratify_group"])
        ):
            df_folds.loc[df_folds.iloc[val_idx].index, "fold"] = fold_num

        return df_folds


def get_data(INPUT_DIR: str) -> WheatData:
    return WheatData(INPUT_DIR)
