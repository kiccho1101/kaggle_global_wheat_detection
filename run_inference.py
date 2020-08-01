# %%
import glob
from src.visualize import imshow_with_bboxes
import numpy as np
import pandas as pd
from src.config import Config
from src.utils import timer, start_mlflow, seed_everything, cp_test_image_to_train
from src.factories import WheatData, WheatDataset, Transforms, Fitter
from src.factories import (
    get_data,
    get_wheat_dataset,
    get_wheat_dataloader,
    get_transforms,
    get_fitter,
    get_tta_transforms,
    get_effdet_eval,
    make_predictions,
    run_wbf,
    inference,
    get_pseudo_train_df,
)
import torch
import torch.utils
import mlflow
import datetime
import matplotlib.pyplot as plt
import cv2
from tqdm.autonotebook import tqdm
from src.types import Boxes

from typing import List, Dict, Any

config = Config(".")

config.n_folds = 0

cp_test_image_to_train(config)
seed_everything(config.seed)
transforms: Transforms = get_transforms()
start_time = datetime.datetime.now().isoformat()

with timer("load raw data"):
    data: WheatData = get_data(config)

cv_num = 0
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

    print("train datanum: ", len(train_loader))
    print("valid datanum: ", len(valid_loader))

    fitter: Fitter = get_fitter(
        cv_num=cv_num, config=config, start_time=start_time, mlflow_on=False
    )

with timer("fit"):
    fitter.fit(train_loader, valid_loader, with_validation=False)

if config.pseudo_labeling:
    with timer("pseudo_labeling"):
        precision, results = fitter.predict_and_evaluate(valid_loader, None, eval=True)
        pseudo_train_df = get_pseudo_train_df(results, config.pseudo_labeling_threshold)
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
        fitter.fit(train_loader, valid_loader, with_validation=False)

with timer("load trained model"):
    model = get_effdet_eval(
        f"{config.WORK_DIR}/output/{start_time}/best-checkpoint_cv{cv_num}.bin"
    ).to(config.device)


def collate_fn(batch):
    return tuple(zip(*batch))


with timer("load test loader"):
    test_dataset: WheatDataset = get_wheat_dataset(
        config.INPUT_DIR,
        np.array(
            [
                path.split("/")[-1][:-4]
                for path in glob.glob(f"{config.INPUT_DIR}/test/*.jpg")
            ]
        ),
        pd.DataFrame(),
        "test",
        transforms.get_test_transforms(),
    )
    test_loader = get_wheat_dataloader(test_dataset, config, "test")

with timer("inference"):
    tta_transforms = get_tta_transforms()
    test_df = inference(model, test_loader, tta_transforms, config)

test_df.to_csv("submission.csv", index=False)
