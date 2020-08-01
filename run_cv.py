# %%
from src.factories.model import get_effdet_eval
import numpy as np
import pandas as pd
from src.config import Config
from src.utils import timer, start_mlflow, seed_everything
from src.factories import WheatData, WheatDataset, Transforms, Fitter
from src.factories import (
    get_data,
    get_wheat_dataset,
    get_wheat_dataloader,
    get_transforms,
    get_fitter,
    get_pseudo_train_df,
)
import torch
import torch.utils
import mlflow
import datetime

from typing import List

config = Config(".")

seed_everything(config.seed)
transforms: Transforms = get_transforms()
start_time = datetime.datetime.now().isoformat()

expriment_id, run_name = start_mlflow(config)
mlflow.log_param("start_time", start_time)


with timer("load raw data"):
    data: WheatData = get_data(config)


precisions: List[float] = []
with timer("CV", mlflow_on=True):
    for cv_num in range(1):
        with timer(f"CV No. {cv_num}"):

            with timer("prepare dataloader and fitter"):
                train_image_ids, train_df, val_image_ids, val_df = data.get_fold(cv_num)

                train_dataset: WheatDataset = get_wheat_dataset(
                    config.INPUT_DIR,
                    train_image_ids,
                    train_df,
                    "train",
                    transforms.get_train_transforms(),
                )
                valid_dataset: WheatDataset = get_wheat_dataset(
                    config.INPUT_DIR,
                    val_image_ids,
                    val_df,
                    "train",
                    transforms.get_valid_transforms(),
                )

                train_loader = get_wheat_dataloader(train_dataset, config, "train")
                valid_loader = get_wheat_dataloader(valid_dataset, config, "valid")

                fitter: Fitter = get_fitter(
                    cv_num=cv_num, config=config, start_time=start_time, mlflow_on=True
                )

            with timer("fit"):
                fitter.fit(train_loader, valid_loader, with_validation=True)

            if config.pseudo_labeling:
                with timer("pseudo_labeling"):
                    precision, results = fitter.predict_and_evaluate(
                        valid_loader, None, eval=True
                    )
                    pseudo_train_df = get_pseudo_train_df(
                        results, config.pseudo_labeling_threshold
                    )
                    train_df = pd.concat([train_df, pseudo_train_df], axis=0)
                    train_image_ids = train_df["image_id"].values
                    train_dataset = get_wheat_dataset(
                        config.INPUT_DIR,
                        train_image_ids,
                        train_df,
                        "train",
                        transforms.get_train_transforms(),
                    )
                    train_loader = get_wheat_dataloader(train_dataset, config, "train")

                with timer("fit again"):
                    fitter.config.n_epochs = config.n_epochs_after_pl
                    fitter.fit(train_loader, valid_loader, with_validation=True)

            with timer("evaluate"):
                precision, results = fitter.predict_and_evaluate(valid_loader)
                precisions.append(precision)

    mlflow.log_metric("precision_avg", np.mean(precisions))
mlflow.end_run()
