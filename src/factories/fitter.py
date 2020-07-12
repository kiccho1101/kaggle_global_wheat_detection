import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import os
import glob

from typing import Dict, Any, Callable


class Fitter:
    def __init__(
        self,
        WORK_DIR: str,
        INPUT_DIR: str,
        model: torch.nn.Module,
        device: torch.device,
        n_epochs: int,
        lr: float,
        loss_fn: Any,
        step_scheduler: bool,
        validation_scheduler: bool,
        scheduler_class: Callable,
        scheduler_params: Dict[str, Any],
        verbose: bool = True,
        verbose_step: int = 10,
    ):
        self.epoch: int = 0
        self.WORK_DIR: str = WORK_DIR
        self.INPUT_DIR: str = INPUT_DIR
        self.start_time: str = datetime.datetime.now().isoformat()
        self.log_path: str = f"{self.WORK_DIR}/output/{self.start_time}"
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        self.best_summary_loss: float = 10 ** 5

        self.model: torch.nn.Module = model
        self.device: torch.device = device
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr
        )
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
        self.n_epochs: int = n_epochs
        self.loss_fn = loss_fn
        self.step_scheduler: bool = step_scheduler
        self.validation_scheduler: bool = validation_scheduler

        self.verbose: bool = verbose
        self.verbose_step: int = verbose_step
        self.log(f"Fitter prepared. Device is {self.device}")

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        for e in range(self.n_epochs):
            if self.verbose:
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

            if self.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def _train_one_epoch(self, train_loader: DataLoader):
        self.model.train()
        summary_loss = self.loss_fn

        start = time.time()
        for step, (images, targets, iamge_ids) in enumerate(train_loader):
            if self.verbose:
                print(
                    f"Train Step {step}/{len(train_loader)}, "
                    + f"summary_loss: {summary_loss.avg:.5f}, "
                    + f"time: {(time.time() - start):.5f}",
                    end="\r",
                )
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            bboxes = [target["bboxes"].to(self.device) for target in targets]
            labels = [target["labels"].to(self.device) for target in targets]

            self.optimizer.zero_grad()

            loss, _, _ = self.model(images, bboxes, labels)
            loss.backward()
            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()

            if self.step_scheduler:
                self.scheduler.step()
        return summary_loss

    def _validation(self, valid_loader: DataLoader):
        self.model.eval()
        summary_loss = self.loss_fn
        start = time.time()
        for step, (images, targets, image_ids) in enumerate(valid_loader):
            if self.verbose:
                if step % self.verbose_step == 0:
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
                boxes = [target["boxes"].to(self.device).float() for target in targets]
                labels = [
                    target["labels"].to(self.device).float() for target in targets
                ]

                loss, _, _ = self.model(images, boxes, labels)
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
        if self.verbose:
            print(message)
        with open(f"{self.log_path}/log.txt", "a+") as logger:
            logger.write(f"{message}\n")


def get_fitter(
    WORK_DIR: str,
    INPUT_DIR: str,
    model: torch.nn.Module,
    device: torch.device,
    n_epochs: int,
    lr: float,
    loss_fn: Any,
    step_scheduler: bool,
    validation_scheduler: bool,
    scheduler_class: Any,
    scheduler_params: Dict[str, Any],
    verbose: bool = True,
    verbose_step: int = 10,
) -> Fitter:
    return Fitter(
        WORK_DIR,
        INPUT_DIR,
        model,
        device,
        n_epochs,
        lr,
        loss_fn,
        step_scheduler,
        validation_scheduler,
        scheduler_class,
        scheduler_params,
        verbose,
        verbose_step,
    )
