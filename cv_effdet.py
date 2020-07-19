# %%
import numpy as np
import pandas as pd
from src.config import Config
from src.utils import timer, start_mlflow
from src.factories import WheatData, WheatDataset, Transforms, Fitter
from src.factories import (
    get_data,
    get_wheat_dataset,
    get_wheat_dataloader,
    get_transforms,
    get_fitter,
    get_average_meter,
    get_effdet,
)
import torch
import torch.utils
import mlflow

from typing import List


config = Config()
transforms: Transforms = get_transforms()

WORK_DIR = "."
INPUT_DIR = f"{WORK_DIR}/input/global-wheat-detection"
effdet_path = f"{WORK_DIR}/input/efficientdet/efficientdet_d5-ef44aea8.pth"


with timer("load raw data"):
    data: WheatData = get_data(INPUT_DIR)


expriment_id, run_name = start_mlflow(config)
losses: List[float] = []
with timer("CV", mlflow_on=True):
    for cv_num in range(3):
        with timer(f"CV No. {cv_num}"):

            with timer("prepare dataloader and fitter"):
                train_image_ids, train_df, val_image_ids, val_df = data.get_fold(cv_num)

                train_dataset: WheatDataset = get_wheat_dataset(
                    INPUT_DIR,
                    train_image_ids,
                    train_df,
                    "train",
                    transforms.get_train_transforms(),
                )
                valid_dataset: WheatDataset = get_wheat_dataset(
                    INPUT_DIR,
                    val_image_ids,
                    val_df,
                    "train",
                    transforms.get_valid_transforms(),
                )

                train_loader = get_wheat_dataloader(train_dataset, config, "train")
                valid_loader = get_wheat_dataloader(valid_dataset, config, "valid")

                device = torch.device("cuda")

                if config.model == "effdet" or config.model == "timm_effdet":
                    model = get_effdet(effdet_path)
                    model.cuda()
                    model.to(device)
                else:
                    model = get_effdet(effdet_path)
                    model.cuda()
                    model.to(device)

                fitter: Fitter = get_fitter(
                    WORK_DIR=WORK_DIR,
                    INPUT_DIR=INPUT_DIR,
                    cv_num=cv_num,
                    model=model,
                    device=device,
                    loss_fn=get_average_meter(),
                    config=config,
                )

            with timer("fit"):
                fitter.fit(train_loader, valid_loader)
                losses.append(fitter.best_summary_loss)

    mlflow.log_metric("cv_loss_avg", np.mean(losses))
    mlflow.end_run()
