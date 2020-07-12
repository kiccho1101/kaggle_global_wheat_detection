import datetime
from pathlib import Path
import torch

from typing import Dict, Any


class Fitter:
    def __init__(
        self,
        WORK_DIR: str,
        INPUT_DIR: str,
        model: torch.nn.Module,
        device: torch.device,
        lr: float,
        scheduler_class: Any,
        scheduler_params: Dict[str, Any],
        verbose: bool = True,
    ):
        self.epoch: int = 0
        self.WORK_DIR: str = WORK_DIR
        self.INPUT_DIR: str = INPUT_DIR
        self.start_time: str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.log_path: str = f"{self.WORK_DIR}/output/{self.start_time}"
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        self.best_summary_loss: float = 10 ** 5

        self.model: torch.nn.Module = model
        self.device: torch.device = device
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr
        )
        self.scheduler: torch.optim.lr_scheduler = scheduler_class(
            self.optimizer, **scheduler_params
        )

        self.verbose: bool = verbose
        self.log(f"Fitter prepared. Device is {self.device}")

    def log(self, message: str):
        if self.verbose:
            print(message)
        with open(f"{self.log_path}/log.txt", "a+") as logger:
            logger.write(f"{message}\n")


def get_fitter(
    WORK_DIR: str,
    INPUT_DIR: str,
    model: torch.nn.Module,
    device: torch.device,
    lr: float,
    scheduler_class: Any,
    scheduler_params: Dict[str, Any],
    verbose: bool = True,
) -> Fitter:
    return Fitter(
        WORK_DIR,
        INPUT_DIR,
        model,
        device,
        lr,
        scheduler_class,
        scheduler_params,
        verbose,
    )
