import torch
import torch.utils
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from src.config import Config


def collate_fn(batch):
    return tuple(zip(*batch))


def get_wheat_dataloader(
    dataset: torch.utils.data.Dataset, config: Config, mode: str
) -> torch.utils.data.DataLoader:
    if mode == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=RandomSampler(dataset),
            pin_memory=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
    if mode == "valid":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            sampler=SequentialSampler(dataset),
            pin_memory=False,
            collate_fn=collate_fn,
        )
    if mode == "test":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
