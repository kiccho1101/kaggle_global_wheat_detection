import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import os
import glob
from tqdm.autonotebook import tqdm
import numpy as np

from src.config import Config

from nptyping import NDArray
from typing import Dict, Any, Callable, List


class Fitter:
    def __init__(
        self,
        WORK_DIR: str,
        INPUT_DIR: str,
        model: torch.nn.Module,
        device: torch.device,
        loss_fn: Any,
        config: Config,
    ):
        self.epoch: int = 0
        self.WORK_DIR: str = WORK_DIR
        self.INPUT_DIR: str = INPUT_DIR
        self.start_time: str = datetime.datetime.now().isoformat()
        self.log_path: str = f"{self.WORK_DIR}/output/{self.start_time}"
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        self.best_summary_loss: float = 10 ** 5

        self.config: Config = config
        self.model: torch.nn.Module = model
        self.loss_fn = loss_fn
        self.device: torch.device = device
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr
        )
        self.scheduler = config.scheduler_class(
            self.optimizer, **config.scheduler_params
        )
        self.log(f"Fitter prepared. Device is {self.device}")

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        for _ in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                timestamp = datetime.datetime.now().isoformat()
                self.log(f"\n{timestamp}\nLR: {lr}")

            start = time.time()
            summary_loss = self._train_one_epoch(train_loader)

            self.log(
                f"[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - start):.5f}"
            )
            self.save(f"{self.log_path}/last-checkpoint.bin")

            start = time.time()
            summary_loss = self._validation(valid_loader)
            self.log(
                f"[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - start):.5f}"
            )

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(
                    f"{self.log_path}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin"
                )
                for path in sorted(
                    glob.glob(f"{self.log_path}/best-checkpoint-*epoch.bin")
                )[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def _train_one_epoch(self, train_loader: DataLoader):
        self.model.train()
        summary_loss = self.loss_fn

        start = time.time()

        for step, (images, targets, _) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            if self.config.verbose:
                print(
                    f"Train Step {step}/{len(train_loader)}, "
                    + f"summary_loss: {summary_loss.avg:.5f}, "
                    + f"time: {(time.time() - start):.5f}",
                )
            images = torch.stack(images)
            images: NDArray[(self.config.batch_size, 3, 512, 512), np.int] = images.to(
                self.device
            ).float()
            bboxes: List[NDArray[(Any, 4), np.int]] = [
                target["bboxes"].to(self.device).float() for target in targets
            ]
            labels: List[NDArray[(Any), np.int]] = [
                target["labels"].to(self.device).float() for target in targets
            ]

            target_res = {"bbox": bboxes, "cls": labels}

            self.optimizer.zero_grad()

            outputs = self.model(images, target_res)
            loss = outputs["loss"]

            loss.backward()
            summary_loss.update(loss.detach().item(), self.config.batch_size)
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss

    def _validation(self, valid_loader: DataLoader):
        self.model.eval()
        summary_loss = self.loss_fn
        start = time.time()
        for step, (images, targets, _) in enumerate(valid_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f"Val Step {step}/{len(valid_loader)}, "
                        + f"summary_loss: {summary_loss.avg:.5f}, "
                        + f"time: {(time.time() - start):.5f}",
                        end="\r",
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                bboxes: List[NDArray[(Any, 4), np.int]] = [
                    target["bboxes"].to(self.device).float() for target in targets
                ]
                labels: List[NDArray[(Any), np.int]] = [
                    target["labels"].to(self.device).float() for target in targets
                ]

                target_res = {"bbox": bboxes, "cls": labels}

                outputs, _, _ = self.model(images, target_res)
                loss = outputs["loss"]

                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def save(self, path: str):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_summary_loss": self.best_summary_loss,
                "epoch": self.epoch,
            },
            path,
        )

    def log(self, message: str):
        if self.config.verbose:
            print(message)
        with open(f"{self.log_path}/log.txt", "a+") as logger:
            logger.write(f"{message}\n")


def get_fitter(
    WORK_DIR: str,
    INPUT_DIR: str,
    model: torch.nn.Module,
    device: torch.device,
    loss_fn: Any,
    config: Config,
) -> Fitter:
    return Fitter(WORK_DIR, INPUT_DIR, model, device, loss_fn, config)
