# %%
import numpy as np
import pandas as pd
from src import Config
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


config = Config()
transforms: Transforms = get_transforms()

WORK_DIR = "."
INPUT_DIR = f"{WORK_DIR}/input/global-wheat-detection"
effdet_path = f"{WORK_DIR}/input/efficientdet/efficientdet_d5-ef44aea8.pth"

data: WheatData = get_data(INPUT_DIR)
train_image_ids, train_df, val_image_ids, val_df = data.get_fold(0)

train_dataset: WheatDataset = get_wheat_dataset(
    INPUT_DIR, train_image_ids, train_df, "train", transforms.get_train_transforms()
)
valid_dataset: WheatDataset = get_wheat_dataset(
    INPUT_DIR, val_image_ids, val_df, "train", transforms.get_valid_transforms()
)

train_loader = get_wheat_dataloader(train_dataset, config, "train")
valid_loader = get_wheat_dataloader(valid_dataset, config, "valid")

device = torch.device("cuda")

model = get_effdet(effdet_path)
model.to(device)

fitter: Fitter = get_fitter(
    WORK_DIR=WORK_DIR,
    INPUT_DIR=INPUT_DIR,
    model=model,
    device=device,
    loss_fn=get_average_meter(),
    config=config,
)

fitter.fit(train_loader, valid_loader)

# %%
