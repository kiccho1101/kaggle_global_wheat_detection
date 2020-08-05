# %%
import numpy as np
import pandas as pd
import glob
from src.config import Config
from src.utils import timer, start_mlflow, seed_everything
from src.factories import WheatData, WheatDataset, Transforms, Fitter
from src.visualize import imshow_with_bboxes
from src.factories import (
    get_data,
    get_wheat_dataset,
    get_wheat_dataloader,
    get_transforms,
    get_fitter,
    get_effdet_train,
    get_effdet_eval,
    get_tta_transforms,
    make_predictions,
    run_wbf,
    get_pseudo_train_df,
    get_results,
    inference,
)
import torch
import torch.utils
from torch.utils.data import DataLoader
import mlflow

import datetime
from tqdm.autonotebook import tqdm
from typing import Any, List


config = Config(".")
start_time = datetime.datetime.now().isoformat()
seed_everything(config.seed)
transforms: Transforms = get_transforms()


data = get_data(config)

cv_num = 0
train_image_ids, train_df, val_image_ids, val_df = data.get_fold(cv_num)

train_dataset = get_wheat_dataset(
    config.INPUT_DIR,
    train_image_ids,
    train_df,
    "train",
    transforms.get_train_transforms(),
)
valid_dataset = get_wheat_dataset(
    config.INPUT_DIR, val_image_ids, val_df, "valid", transforms.get_valid_transforms(),
)

train_loader = get_wheat_dataloader(train_dataset, config, "train")
valid_loader = get_wheat_dataloader(valid_dataset, config, "valid")


# %%
checkpoint_path = f"{config.WORK_DIR}/input/wheat-effdet5-fold0-best-checkpoint/fold0-best-all-states.bin"
model = get_effdet_eval(checkpoint_path).to(config.device).eval()

test_dataset = get_wheat_dataset(
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

results = get_results(test_loader, model, config)
pseudo_df = get_pseudo_train_df(results, 0.05)

# %%
pseudo_dataset = get_wheat_dataset(
    config.INPUT_DIR,
    pseudo_df["image_id"].unique(),
    pseudo_df,
    "train",
    transforms.get_train_transforms(),
)
image, target, image_id = pseudo_dataset[2]
imshow_with_bboxes(image, target["bboxes"], bbox_type="yxyx")

# %%
train_df = pd.concat([train_df, pseudo_df], axis=0)
train_image_ids = train_df["image_id"].unique()
train_dataset = get_wheat_dataset(
    config.INPUT_DIR,
    train_image_ids,
    train_df,
    "train",
    transforms.get_train_transforms(),
)
train_loader = get_wheat_dataloader(train_dataset, config, "train")

fitter = get_fitter(cv_num=cv_num, config=config, start_time=start_time)
fitter.set_train_model(checkpoint_path)
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
