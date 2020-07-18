import torch
import torch.utils
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from src.config import Config


def collate_fn(batch):
    return tuple(zip(*batch))


def get_wheat_dataloader(
    dataset: torch.utils.data.Dataset, config: Config, mode: str
) -> torch.utils.data.DataLoader:
    dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader()
    if mode == "train":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=RandomSampler(dataset),
            pin_memory=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
    if mode == "valid":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            sampler=SequentialSampler(dataset),
            pin_memory=False,
            collate_fn=collate_fn,
        )
    return dataloader
