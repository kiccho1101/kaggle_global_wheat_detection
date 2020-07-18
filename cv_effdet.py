# %%
import numpy as np
import pandas as pd
from src import Config
from src.factories import WheatData, WheatDataset, Transforms, Fitter
from src.factories import (
    get_data,
    get_wheat_dataset,
    get_transforms,
    get_fitter,
    get_average_meter,
    get_effdet,
)
import torch
import torch.utils
from torch.utils.data.sampler import SequentialSampler, RandomSampler


config = Config()
transforms: Transforms = get_transforms()

WORK_DIR = "."
INPUT_DIR = f"{WORK_DIR}/input/global-wheat-detection"
effdet_path = f"{WORK_DIR}/input/efficientdet/efficientdet_d5-ef44aea8.pth"

data: WheatData = get_data(INPUT_DIR)
train_image_ids, train_df, val_image_ids, val_df = data.get_fold(0)

train_ds: WheatDataset = get_wheat_dataset(
    INPUT_DIR, train_image_ids, train_df, "train", transforms.get_train_transforms()
)
valid_ds: WheatDataset = get_wheat_dataset(
    INPUT_DIR, val_image_ids, val_df, "train", transforms.get_valid_transforms()
)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config.batch_size,
    sampler=RandomSampler(train_ds),
    pin_memory=True,
    num_workers=config.num_workers,
    collate_fn=collate_fn,
)
valid_loader = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False,
    sampler=SequentialSampler(valid_ds),
    pin_memory=False,
    collate_fn=collate_fn,
)

device = torch.device("cuda")

model = get_effdet(effdet_path)
model.to(device)

fitter: Fitter = get_fitter(
    WORK_DIR=WORK_DIR,
    INPUT_DIR=INPUT_DIR,
    model=model,
    device=device,
    n_epochs=config.n_epochs,
    lr=config.lr,
    loss_fn=get_average_meter(),
    step_scheduler=config.step_scheduler,
    validation_scheduler=config.validation_scheduler,
    scheduler_class=config.SchedulerClass,
    scheduler_params=config.scheduler_params,
    verbose=config.verbose,
    verbose_step=config.verbose_step,
)

fitter.fit(train_loader, valid_loader)
